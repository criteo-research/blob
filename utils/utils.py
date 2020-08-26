import pandas as pd
from pylab import *
from torch.utils.data import Dataset

from utils.utils_vae import block_torch_update


import torch
import torch.nn.functional as F
import torch.nn as nn


def to_categorical(y, num_classes=None, dtype='float32'): # from keras
    y = np.array(y.cpu(), dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def shrink_wrap(sess_array, sess_mask, truncate=True):

    if truncate:
        max_sess = int(max(sess_mask.sum(2)))
        sess_array = sess_array[:,:,:max_sess]
        sess_mask = sess_mask[:,:,:max_sess]

    sess_array = sess_array[:,0,:].long()     
    sess_mask = sess_mask[:,0,:]

    return sess_array, sess_mask


def cov2corr(A):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    return A


class PandasDataset(Dataset):
    """Custom dataset object to load Reco-Gym organic only from logged"""

    def __init__(self, P, rawdata, testing=False, prodid2internal=None, drop=True, max_session=20):
        self.max_session = max_session
        
        # Drop all the bandit events
        rawdata = rawdata[rawdata.z != 'bandit'].copy()
        rawdata['v']= rawdata['v'].astype('int64')

        self.testing = testing
        self.rawdata = rawdata
        self.rawdata.loc[:,'internalid'] = self.rawdata['v'].copy()

        # remove sessions of length one
        self.rawdata = self.rawdata.loc[self.rawdata.duplicated(subset='u', keep=False), :]        
        print("Sessions of length one removed")

        # create internal id of sequential product numbers
        self.unique_prod = pd.unique(self.rawdata['v'])

        print("Internal IDs Created")

        # compute max sequence length stuff
        self.unique_sess = pd.unique(self.rawdata['u'])

        # create index dict:
        print("Building session lookup map... ",)
        sessid = self.rawdata['u'].values
        sessid[sessid==0] = max(sessid)+1 # convolve doesn't handle zero correctly 
        self.boundaries = np.arange(len(sessid)+1)
        self.boundaries = self.boundaries[np.convolve(sessid,[1,-1]) != 0]
        assert len(self.boundaries) - 1 == len(self.unique_sess)
        print("Done")

    def process_sessions(self, index):

        sessids = int(self.unique_sess[index])
        batch_df = self.rawdata[self.boundaries[index]:self.boundaries[index+1]]
        views = batch_df.internalid.values

        # Cap the max session length 
        if len(views) > self.max_session:
             views = views[0:self.max_session]

        sess_array = zeros((1, self.max_session))
        sess_mask = zeros((1, self.max_session))

        if not self.testing:
            sess_array[0, 0:len(views)] = views 
            sess_mask[0,0:len(views)]=  1        
            sess_mask = torch.Tensor(sess_mask)

            return (sess_array, sess_mask, np.array(sessids)-1)
        else: 
            sess_array[0, 0:len(views[:-1])] = views[:-1]
            sess_mask[0,0:len(views[:-1])]=  1        
            sess_mask = torch.Tensor(sess_mask)

            return (sess_array, sess_mask, np.array(sessids)-1, views[-1])


    def __len__(self):
        return len(pd.unique(self.rawdata.u))

    def __getitem__(self, index):
        """Process a single element from the dataset"""

        sample = self.process_sessions(index)
        return sample


# unlike the previous dataset produce one record per
# bandit event instead of one record per session
class PandasDatasetBandit(Dataset):
    """Custom dataset object to load Reco-Gym organic only from logged"""

    def __init__(self, rawdata, max_session=20):
        self.max_session = max_session

        history = []
        action = []
        reward = []
        for u in range(int(max(rawdata.u))):
            df = rawdata[rawdata['u']==u]
            bandit_index = np.array(df.t[np.array(df.z)=='bandit'],dtype=int)
            c = np.array(df.c)
            a = np.array(df.a)
            for ii in range(len(bandit_index)):
                temp = df[0:bandit_index[ii]]
                history.append(np.array(temp[temp.z=='organic'].v,dtype=np.int32))
                action.append(int(a[bandit_index[ii]]))
                reward.append(int(c[bandit_index[ii]]))
        self.history = history
        self.action = action
        self.reward = reward

    def __len__(self):
        return len(self.history)

    def __getitem__(self, index):
        """get a single bandit eventt"""
        sess_array = zeros((1, self.max_session))
        sess_mask = zeros((1, self.max_session))
        views = self.history[index]
        if len(views) > self.max_session:
            views = views[0:self.max_session]
        sess_array[0, 0:len(views)] = views 
        sess_mask[0,0:len(views)]=  1        
        sess_mask = torch.LongTensor(sess_mask)

        return (sess_array, sess_mask, self.action[index], self.reward[index])


class Net(nn.Module):
    def __init__(self,K, h1, h2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(K, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, K)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def linear_auto_encoder(sess_array, sess_mask, MB, K, WW_muq, WW_inv_Sigmaq_diag, BB_muq, BB_inv_Sigmaq_diag):
    CW_muq = (WW_muq[sess_array,:] * sess_mask[:,:,None]).sum((1)).reshape(MB,K,1)
    CW_inv_Sigmaq_diag = (WW_muq[sess_array,:] * sess_mask[:,:,None]).sum((1)) 
    p_muq = CW_muq + BB_muq
    p_inv_Sigmaq_diag = F.softplus(CW_inv_Sigmaq_diag + BB_inv_Sigmaq_diag)
    return p_muq, p_inv_Sigmaq_diag


def VB_EM(sess_array, sess_mask, MB, P, K, p_Psi, p_rho, device, cpu_device, Niter=100):
    p_C = to_count_rep(sess_array, sess_mask, MB, P, cpu_device)
    p_xi = torch.Tensor(kron(rand(P,1),ones((1,MB))).T).to(cpu_device)
    p_a = torch.Tensor([1. for _ in range(MB)]).to(cpu_device)
    for _ in range(Niter):        
        p_inv_Sigmaq, p_muq, p_a, p_xi = block_torch_update(p_C, p_a, p_xi, p_Psi.to(cpu_device), p_rho.to(cpu_device), cpu_device)
    p_inv_Sigmaq_diag = torch.stack([torch.diag(p_inv_Sigmaq[ii,:,:]) for ii in range(MB)],0)
    return p_muq.to(device), p_inv_Sigmaq_diag.to(device), p_xi.reshape(MB,P,1).to(device), p_a.reshape(MB,1,1).to(device)


def to_count_rep(sess_array, sess_mask, MB, P, device):
    SL = sess_mask.sum(1)
    return torch.tensor(vstack([to_categorical(sess_array[ii,:int(SL[ii])],P).sum(0) for ii in range(len(SL))]).reshape(MB,P,1)).to(device)
