import os
import sys
import pdb
import datetime

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

import torch
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import PandasDataset, shrink_wrap, cov2corr, to_categorical, linear_auto_encoder, VB_EM
from utils.utils_vae import unintegrated_lower_bound_diagonal_rao_blackwell_v


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class RecoModelRTAE():

    def __init__(self, args):
        super(RecoModelRTAE, self).__init__()

        self.cpu_device = torch.device('cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = self.cpu_device

        self.args = args
        self.run_name = f"{args['algorithm']}_{args['P']}_{args['batch_size']}_{args['K']}_{datetime.datetime.now()}"
        self.use_em = args['use_em']

        # Dataset stuff
        if isinstance(args['dataset'], str):

            self.data_hash = None

            fn_train = args['dataset'] + '/train.csv'
            fn_test = args['dataset'] + '/test.csv'
            self.dataset = PandasDataset(args['P'], pd.read_csv(fn_train), testing=True)
            self.dataset_test = PandasDataset(args['P'], pd.read_csv(fn_test), testing=True)

        else: # support the direct insertion of a dataframe into the dataset
            self.dataset = PandasDataset(args['P'], args['dataset'], testing=True)
            self.data_hash = hash_pandas_object(args['dataset'].v).sum()

    def do_training(self):
        
        writer = SummaryWriter(f"/tmp/tensorboard/ban_bands_rpt_{self.run_name}")

        if self.args['cudavis'] != None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args['cudavis']

        self.P = self.args['P']
        self.K = self.args['K']  
        BS = self.args['batch_size']
        self.rec_at_k = self.args['rcatk']
       
        self.p_Psi = torch.rand(self.P, self.K).to(self.device)
        self.p_rho = torch.rand(self.P, 1).to(self.device)
        self.p_Psi.requires_grad=True
        self.p_rho.requires_grad=True
        param_list = [self.p_Psi, self.p_rho]

        self.WW_muq = torch.rand(self.P, self.K).to(self.device)
        self.WW_inv_Sigmaq_diag = torch.rand(self.P, self.K).to(self.device)
        self.BB_muq=torch.rand(self.K, 1).to(self.device)  
        self.BB_inv_Sigmaq_diag=torch.rand(self.K).to(self.device)        
        self.WW_muq.requires_grad = True
        self.WW_inv_Sigmaq_diag.requires_grad=True
        self.BB_muq.requires_grad=True
        self.BB_inv_Sigmaq_diag.requires_grad=True
        param_list = [self.p_Psi, self.p_rho, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag]

        dataloader = DataLoader(self.dataset, batch_size=BS, shuffle=False, num_workers=6)

        optimizer = optim.RMSprop(param_list, lr=self.args['lr'])

        bar_form = '{l_bar}{bar} | {n_fmt}/{total_fmt} [{remaining} {postfix}]'

        print('Training the Model!')

        try:
            os.mkdir('../weights')
        except:
            pass

        training_counter = 0
        logging_freq = 100
        mean_loss = 0
        snap = str(datetime.datetime.now())

        # Loop over the number of epochs
        for epoch in range(self.args['num_epochs']):

            # Training ------------------------------------------------------------------------------
            with tqdm(total=len(dataloader), desc='Epochs {}/{}'.format(epoch+1, self.args['num_epochs']), bar_format=bar_form) as pbar:

                for train in dataloader:

                    sess_array, sess_mask, sessids, lastv = train
                    lastv = lastv.to(self.device)
                    sess_array, sess_mask = shrink_wrap(sess_array, sess_mask)
                    MB = sess_array.shape[0]
                    sess_mask = sess_mask.to(self.device)
                    sess_array = sess_array.to(self.device)
        
                    optimizer.zero_grad()
                
                    self.p_muq, self.p_inv_Sigmaq_diag = linear_auto_encoder(sess_array, sess_mask, MB, self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag)
                    self.epsilon = torch.randn(MB, self.K).to(self.device)
                    self.p_omega = (torch.sqrt(self.p_inv_Sigmaq_diag) * self.epsilon).reshape(MB, self.K, 1) + self.p_muq
                    loss = -torch.sum(unintegrated_lower_bound_diagonal_rao_blackwell_v(sess_array, sess_mask, self.p_Psi, self.p_rho, self.p_omega, self.p_inv_Sigmaq_diag, self.p_muq))

                    # perform back prop
                    loss.backward()
                    optimizer.step()

                    mean_loss += loss.item()

                    if (training_counter+1) % logging_freq == 0:
                        writer.add_scalar('data/loss', mean_loss/logging_freq, training_counter/logging_freq)
                        mean_loss = 0
                    
                    training_counter += 1
                    pbar.update(1)
                

        writer.close()
        print('Training Complete!')
        if 'weight_path' in self.args:
            torch.save(param_list, self.args['weight_path'])
            print('saving to ' + self.args['weight_path'])

            fp = open(self.args['weight_path'] + '_hash','w')
            fp.write(str(self.data_hash))
            fp.close()

    def create_user_embedding(self, sess_array, sess_mask):

        MB = sess_array.shape[0]
        sess_mask = sess_mask.to(self.device)
        sess_array = sess_array.to(self.device)
        p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array, sess_mask, MB, self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag)

        return p_muq, p_inv_Sigmaq_diag

    def next_item_prediction(self, sess_array, sess_mask):

        MB = sess_array.shape[0]
        sess_mask = sess_mask.to(self.device)
        sess_array = sess_array.to(self.device)
        if self.use_em:
            p_muq, p_inv_Sigmaq_diag, p_xi, p_a = VB_EM(sess_array, sess_mask, MB, self.P, self.K, self.p_Psi, self.p_rho, self.device, self.cpu_device, Niter=100)
        else:
            p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array.long(), sess_mask.float(), MB, self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag)
        logits = torch.matmul(self.p_Psi, p_muq) + self.p_rho

        return logits

class RecoModelItemKNN():
    def __init__(self, args):
        super(RecoModelItemKNN, self).__init__()
        self.cpu_device = torch.device('cpu')
        self.device = self.cpu_device

        self.args = args
        self.run_name = f"{args['algorithm']}_{args['P']}_{args['batch_size']}_{args['K']}_{datetime.datetime.now()}"
        self.args['num_epochs'] = 1

        self.co_counts = np.eye(self.args['P'])

        # Dataset stuff
        if isinstance(args['dataset'], str):
            fn_train = args['dataset'] + '/train.csv'
            fn_test = args['dataset'] + '/test.csv'
            self.dataset = PandasDataset(args['P'], pd.read_csv(fn_train), testing=True)
            self.dataset_test = PandasDataset(args['P'], pd.read_csv(fn_test), testing=True)

        else: # support the direct insertion of a dataframe into the dataset
            self.dataset = PandasDataset(args['P'], args['dataset'], testing=True)

    def do_training(self):
        dataloader = DataLoader(self.dataset, batch_size=self.args['batch_size_test'], shuffle=False, num_workers=6, drop_last=True)

        metric_l = []
        count = 0

        bar_form = '{l_bar}{bar} | {n_fmt}/{total_fmt} [{remaining} {postfix}]'

        with tqdm(total=len(dataloader), desc='Epochs {}/{}'.format(1, self.args['num_epochs']), bar_format=bar_form) as pbar:

            for train in dataloader:
                sess_array, sess_mask, sessids, lastv = train
                lastv = lastv.to(self.device)
                sess_array, sess_mask = shrink_wrap(sess_array, sess_mask)
                MB = sess_array.shape[0]
                sess_mask = sess_mask.to(self.device)
                sess_array = sess_array.to(self.device)

                SL = sess_mask.sum(1).long()
                ss = np.vstack([to_categorical(sess_array[ii,SL[ii]-1],self.args['P']) for ii in range(sess_array.shape[0])]) + np.vstack([to_categorical(lv,self.args['P']) for lv in lastv])
                self.co_counts += np.matmul(ss.T,ss)

                pbar.update(1)

        self.corr = torch.tensor(cov2corr(self.co_counts))


    # for the most likely item not to be the most recent you can't look only the most recent item.
    def next_item_prediction(self, sess_array, sess_mask):
        MB = sess_array.shape[0]
        sess_mask = sess_mask.to(self.device)
        sess_array = sess_array.to(self.device)
        SL = sess_mask.sum(1).long()
        return torch.stack([self.corr[:,sess_array[ii,0:SL[ii].long()].long()].mean(1).reshape(1,self.args['P']) for ii in range(len(SL))]).reshape(MB,self.args['P'])