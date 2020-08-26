import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.math import softplus as sp

import os
import sys
import datetime

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object

import torch
from torch import optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.utils import PandasDataset, PandasDatasetBandit
from utils.utils import shrink_wrap, linear_auto_encoder, VB_EM
from utils.utils_vae import unintegrated_lower_bound_diagonal_rao_blackwell_v


def KL_multivariate(means, logstds, means_0, stds_0):
    """
    KL in the special case where the target distribution 
    is factorized and the prior is any multivariate gaussian
    """
    
    dets =  2.0*tf.reduce_sum(tf.log(stds_0) - logstds) - tf.cast(means.shape[0], tf.float32)
    norm_trace = tf.reduce_sum(((means - means_0)**2 + tf.exp(logstds)**2)/stds_0**2)
    
    KL = 0.5*(dets + norm_trace)
    
    return KL

# The Matrix Normal Variational Approximation :
class RecoModelRTAEWithBanditTF():

    def __init__(self, args):
        super(RecoModelRTAEWithBanditTF, self).__init__()

        self.cpu_device = torch.device('cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #FIXME FIXME FIXME GPU off atm
        self.device = self.cpu_device

        self.args = args
        self.run_name = f"{args['algorithm']}_{args['P']}_{args['batch_size']}_{args['K']}_{datetime.datetime.now()}"
        self.use_em = args['use_em']

        self.do_organic_training = True

        self.P = self.args['P']
        self.K = self.args['K']  

        # Dataset stuff
        if isinstance(args['dataset'], str):
            fn_train = args['dataset'] + '/train.csv'
            fn_test = args['dataset'] + '/test.csv'
            df_train = pd.read_csv(fn_train)
            self.dataset = PandasDataset(args['P'], df_train, testing=True)
            self.dataset_test = PandasDataset(args['P'], pd.read_csv(fn_test), testing=True)

            self.data_hash = None

            print('creating bandit dataset')
            self.dataset_bandit = PandasDatasetBandit(df_train)
            print('done')

        else: # support the direct insertion of a dataframe into the dataset
            self.dataset = PandasDataset(args['P'], args['dataset'], testing=True)

            self.data_hash = hash_pandas_object(args['dataset'].v).sum()

            if 'organic_weights' in self.args:
                fp = open(self.args['organic_weights']+ '_hash')
                hh = int(fp.readline())
                fp.close()
                if self.data_hash != hh:
                    print('the weights seem to be trained off a different dataset')
                    sys.exit(1)
                self.p_Psi, self.p_rho, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag = torch.load(self.args['organic_weights'])

                if self.p_Psi.shape[0] != self.P:
                    print('the saved weights have the wrong P')
                    sys.exit(1)
                if self.p_Psi.shape[1] != self.K:
                    print('the saved weights have the wrong K')
                    sys.exit(1)
                self.do_organic_training = False
                print('successfully loaded organic weights')

        

            print('creating bandit dataset')
            self.dataset_bandit = PandasDatasetBandit(args['dataset'])
            print('done')

    def do_training(self):
        
        writer = SummaryWriter(f"/tmp/tensorboard/ban_bands_rpt_{self.run_name}")

        if self.args['cudavis'] != None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args['cudavis']

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
        logging_freq = 10
        mean_loss = 0
        snap = str(datetime.datetime.now())


        if self.do_organic_training:
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
                    
                        self.p_muq, self.p_inv_Sigmaq_diag = linear_auto_encoder(sess_array, sess_mask, MB, 
                                                                                self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, 
                                                                                self.BB_muq, self.BB_inv_Sigmaq_diag)
                        self.epsilon = torch.randn(MB, self.K).to(self.device)
                        self.p_omega = (torch.sqrt(self.p_inv_Sigmaq_diag) * self.epsilon).reshape(MB, self.K, 1) + self.p_muq
                        loss = -torch.sum(unintegrated_lower_bound_diagonal_rao_blackwell_v(sess_array, sess_mask, self.p_Psi, self.p_rho, 
                                                                                            self.p_omega, self.p_inv_Sigmaq_diag, self.p_muq))

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
            print('Organic Training Complete!')


            try:
                os.mkdir('organic_params')
            except:
                pass

            try:
                os.mkdir('organic_params/' + self.run_name.replace('/','|'))
            except:
                pass

            organic_params = [self.p_Psi, self.p_rho, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag]

            torch.save(organic_params, 'organic_params/' + self.run_name.replace('/','|') + '/' + 'weights.pt')
            fp = open('organic_params/' + self.run_name.replace('/','|') + '/' + 'hash.pt','w')
            fp.write(str(self.data_hash))
            fp.close()

        self.p_Psi.requires_grad=False
        self.p_rho.requires_grad=False
        self.WW_muq.requires_grad = False
        self.WW_inv_Sigmaq_diag.requires_grad=False
        self.BB_muq.requires_grad=False
        self.BB_inv_Sigmaq_diag.requires_grad=False

        print('Starting bandit training')

        #Necessary values
        
        dataloader_bandit = DataLoader(self.dataset_bandit, 
                                       batch_size=BS, 
                                       shuffle=True, 
                                       num_workers=6)
        K, P = self.K, self.P
        N = len(dataloader_bandit.dataset)

        #Psi for the location of the MN
        Psi_loc = self.p_Psi.detach().numpy()
        
        # define the Psi cov : 
        Psi_cov = Psi_loc
        if self.args['norm'] : Psi_cov /= np.linalg.norm(Psi_cov, axis = 0, keepdims=True)
        
        #Cholesky decomposition
        cov_k = Psi_cov.T.dot(Psi_cov)/self.P
        L = np.linalg.cholesky(cov_k)
        
       # tf.reset_default_graph()
        
        tf_psi_loc = tf.constant(Psi_loc)
        tf_psi_cov = tf.constant(Psi_cov)
        tf_L = tf.constant(L)

        #DEFINING PRIORS : 
        #Hyperparameters
        s_zeta = 1.0
        kappa_m = 0.0

        wa_m = self.args['wa_m']
        wb_m = self.args['wb_m']
        bias_m = self.args['wc_m']

        wa_s = self.args['wa_s']
        wb_s = self.args['wb_s']
        bias_s = self.args['wc_s']
        kappa_s = self.args['kappa_s']

        #W prior 0,1
        zeta_0_mean = tf.zeros([K, K])
        zeta_0_std1 = s_zeta*tf.ones([K,1])
        zeta_0_std2 = tf.ones([K,1])
        zeta_0_std = tf.reshape(tf.reshape(zeta_0_std1, [K,1])*tf.reshape(zeta_0_std2, [1,K]), [K**2])

        #Bias prior 
        bias_0_mean = tf.constant([bias_m])
        bias_0_std = tf.constant([bias_s])

        #W_a prior
        wa_0_mean = tf.constant([wa_m])
        wa_0_std = tf.constant([wa_s])

        #W_b prior
        wb_0_mean = tf.constant([wb_m])
        wb_0_std = tf.constant([wa_s])
        
        #Kappa prior 
        kappa_0_mean = tf.zeros([P,1])
        kappa_0_std = kappa_s*tf.ones([P,1])
        
        X = tf.placeholder(dtype=tf.float32, shape = [None, K])
        Y = tf.placeholder(dtype=tf.float32, shape = [None, 1])
        A = tf.placeholder(dtype=tf.int32, shape = [None,])

        #Zeta posterior
        zeta_means = tf.Variable(zeta_0_mean, dtype=tf.float32)
        zeta_logstd1 = tf.Variable(tf.log(zeta_0_std1), dtype=tf.float32)
        zeta_logstd2 = tf.Variable(tf.log(zeta_0_std2), dtype=tf.float32)

        #Bias posterior
        bias_means = tf.Variable(bias_0_mean, dtype=tf.float32)
        bias_logstd = tf.Variable(tf.log(bias_0_std), dtype=tf.float32)

        #wa posterior
        wa_means = tf.Variable(wa_0_mean, dtype=tf.float32)
        wa_logstd = tf.Variable(tf.log(wa_0_std), dtype=tf.float32)

        #wb posterior
        wb_means = tf.Variable(wb_0_mean, dtype=tf.float32)
        wb_logstd = tf.Variable(tf.log(wb_0_std), dtype=tf.float32)

        #Kappa posterior
        kappa_means = tf.Variable(kappa_0_mean, dtype=tf.float32)
        kappa_logstd = tf.Variable(tf.log(kappa_0_std), dtype=tf.float32)

        #Useful tensors :
        L_omega_u = tf.matmul(X, tf_L)
        Psi_a_u_loc = tf.gather(tf_psi_loc, A, axis = 0)
        Psi_a_u = tf.gather(tf_psi_cov, A, axis = 0)
        kappa_a_u = tf.gather(kappa_means, A, axis = 0)
        kappa_logstd_a_u = tf.gather(kappa_logstd, A, axis = 0)

        R1 = tf.matmul(Psi_a_u**2, tf.exp(2*zeta_0_std1))
        R2 = tf.matmul(L_omega_u**2, tf.exp(2*zeta_0_std2))

        #ORGANIC :
        wa_noise = tf.random_normal([tf.shape(X)[0], 1])
        pred_org = sp(wa_means + tf.exp(wa_logstd)*wa_noise)*tf.reduce_sum(Psi_a_u_loc*X, axis = 1, keepdims=True) 

        #BANDIT :
        wb_noise = tf.random_normal([tf.shape(X)[0], 1])
        band_noise = tf.random_normal([tf.shape(X)[0], 1])
        pred_band =  sp(wb_means + tf.exp(wb_logstd)*wb_noise)*(tf.reduce_sum(tf.matmul(Psi_a_u, zeta_means)*L_omega_u, axis = 1, keepdims=True) + tf.sqrt(R1*R2)*band_noise)

        #Bias :
        bias_noise = tf.random_normal([tf.shape(X)[0], 1])
        pred_bias = bias_means + kappa_a_u + tf.sqrt(tf.exp(2*bias_logstd) + tf.exp(2*kappa_logstd_a_u))*bias_noise

        predictions = pred_org + pred_band + pred_bias

        pred_distribution = tfd.Bernoulli(logits=predictions)
        neg_log_prob = -tf.reduce_mean(pred_distribution.log_prob(Y))

        zeta_logstd = tf.reshape((tf.reshape(zeta_logstd1, [K,1]) + tf.reshape(zeta_logstd2, [1,K])), [K**2])
        kl_zeta = KL_multivariate(tf.reshape(zeta_means, [K**2]), zeta_logstd, tf.reshape(zeta_0_mean, [K**2]), zeta_0_std)
        kl_bias = KL_multivariate(bias_means, bias_logstd, bias_0_mean, bias_0_std)
        kl_wa = KL_multivariate(wa_means, wa_logstd, wa_0_mean, wa_0_std)
        kl_wb = KL_multivariate(wb_means, wb_logstd, wb_0_mean, wb_0_std)
        kl_kappa = KL_multivariate(kappa_means, kappa_logstd, kappa_0_mean, kappa_0_std)

        kl_div =  kl_bias + kl_wa + kl_wb + kl_zeta + kl_kappa

        neg_ELBO = neg_log_prob + kl_div/N        # Use ADAM optimizer w/ -ELBO loss
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(neg_ELBO)

        # Initialization op
        init_op = tf.global_variables_initializer()
        

        # Loop over the number of epochs
        
        # Run the training session
        with tf.Session() as sess:
            sess.run(init_op)
            for epoch in range(self.args['num_epochs_bandit']):
                # Training ------------------------------------------------------------------------------
                with tqdm(total=len(dataloader_bandit), desc='Epochs {}/{}'.format(epoch+1, self.args['num_epochs_bandit']), bar_format=bar_form) as pbar:
                    
                    for train_bandit in dataloader_bandit:
                        
                        sess_array, sess_mask, action, reward = train_bandit
                        sess_array, sess_mask = shrink_wrap(sess_array, sess_mask)
                        MB = sess_array.shape[0]
                        sess_mask = sess_mask.to(self.device)
                        sess_array = sess_array.to(self.device)

                        if self.use_em:
                            p_muq, p_inv_Sigmaq_diag, p_xi, p_a = VB_EM(sess_array.long(), sess_mask.float(), MB, self.P, self.K, self.p_Psi, 
                                                                        self.p_rho, self.device, self.cpu_device, Niter=100)
                        else:
                            p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array.long(), sess_mask.float(), MB, 
                                                                           self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, 
                                                                           self.BB_muq, self.BB_inv_Sigmaq_diag)

                        bx = p_muq.reshape(MB, self.K).detach().numpy()
                        by = reward.numpy().reshape(-1, 1)
                        ba = action.numpy()
                        
                        sess.run(train_op, feed_dict={X:bx, Y:by, A:ba})

                        pbar.update(1)

            print('bandit training complete')
            
            self.p_beta = torch.Tensor(sess.run(sp(wa_means)*tf_psi_loc + sp(wb_means)*tf.matmul(tf.matmul(tf_psi_cov, zeta_means), tf.transpose(tf_L))))
            self.p_kappa = torch.Tensor(sess.run(kappa_means)[:,0])
            self.wa_lr = sess.run(wa_means)
            self.wb_lr = sess.run(wb_means)


    def create_user_embedding(self, sess_array, sess_mask):

        MB = sess_array.shape[0]
        sess_mask = sess_mask.to(self.device)
        sess_array = sess_array.to(self.device)
        p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array, sess_mask, MB, self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag)

        return p_muq, p_inv_Sigmaq_diag



    def bandit_prediction(self, sess_array, sess_mask):
        MB = sess_array.shape[0]
        sess_mask = sess_mask.to(self.device)
        sess_array = sess_array.to(self.device)
        if self.use_em:
            p_muq, p_inv_Sigmaq_diag, p_xi, p_a = VB_EM(sess_array, sess_mask, MB, self.P, self.K, self.p_Psi, 
                                                        self.p_rho, self.device, self.cpu_device, Niter=100)
        else:
            p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array.long(), sess_mask, MB, self.K, 
                                                           self.WW_muq, self.WW_inv_Sigmaq_diag, 
                                                           self.BB_muq, self.BB_inv_Sigmaq_diag)
        user_embedding = p_muq.reshape(MB, self.K)
        logits = torch.matmul(user_embedding, self.p_beta.t()) + self.p_kappa
        return logits

# The Gaussian Variational Approximation :
class RecoModelRTAEWithBanditTF_Full():

    def __init__(self, args):
        super(RecoModelRTAEWithBanditTF_Full, self).__init__()

        self.cpu_device = torch.device('cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #FIXME FIXME FIXME GPU off atm
        self.device = self.cpu_device

        self.args = args
        self.run_name = f"{args['algorithm']}_{args['P']}_{args['batch_size']}_{args['K']}_{datetime.datetime.now()}"
        self.use_em = args['use_em']

        self.do_organic_training = True

        self.P = self.args['P']
        self.K = self.args['K']  

        # Dataset stuff
        if isinstance(args['dataset'], str):
            fn_train = args['dataset'] + '/train.csv'
            fn_test = args['dataset'] + '/test.csv'
            df_train = pd.read_csv(fn_train)
            self.dataset = PandasDataset(args['P'], df_train, testing=True)
            self.dataset_test = PandasDataset(args['P'], pd.read_csv(fn_test), testing=True)

            self.data_hash = None

            print('creating bandit dataset')
            self.dataset_bandit = PandasDatasetBandit(df_train)
            print('done')

        else: # support the direct insertion of a dataframe into the dataset
            self.dataset = PandasDataset(args['P'], args['dataset'], testing=True)

            self.data_hash = hash_pandas_object(args['dataset'].v).sum()

            if 'organic_weights' in self.args:
                fp = open(self.args['organic_weights']+ '_hash')
                hh = int(fp.readline())
                fp.close()
                if self.data_hash != hh:
                    print('the weights seem to be trained off a different dataset')
                    sys.exit(1)
                self.p_Psi, self.p_rho, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag = torch.load(self.args['organic_weights'])

                if self.p_Psi.shape[0] != self.P:
                    print('the saved weights have the wrong P')
                    sys.exit(1)
                if self.p_Psi.shape[1] != self.K:
                    print('the saved weights have the wrong K')
                    sys.exit(1)
                self.do_organic_training = False
                print('successfully loaded organic weights')

        

            print('creating bandit dataset')
            self.dataset_bandit = PandasDatasetBandit(args['dataset'])
            print('done')

    def do_training(self):
        
        writer = SummaryWriter(f"/tmp/tensorboard/ban_bands_rpt_{self.run_name}")

        if self.args['cudavis'] != None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.args['cudavis']

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
        logging_freq = 10
        mean_loss = 0
        snap = str(datetime.datetime.now())


        if self.do_organic_training:
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
                    
                        self.p_muq, self.p_inv_Sigmaq_diag = linear_auto_encoder(sess_array, sess_mask, MB, 
                                                                                self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, 
                                                                                self.BB_muq, self.BB_inv_Sigmaq_diag)
                        self.epsilon = torch.randn(MB, self.K).to(self.device)
                        self.p_omega = (torch.sqrt(self.p_inv_Sigmaq_diag) * self.epsilon).reshape(MB, self.K, 1) + self.p_muq
                        loss = -torch.sum(unintegrated_lower_bound_diagonal_rao_blackwell_v(sess_array, sess_mask, self.p_Psi, self.p_rho, 
                                                                                            self.p_omega, self.p_inv_Sigmaq_diag, self.p_muq))

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
            print('Organic Training Complete!')


            try:
                os.mkdir('organic_params')
            except:
                pass

            try:
                os.mkdir('organic_params/' + self.run_name.replace('/','|'))
            except:
                pass

            organic_params = [self.p_Psi, self.p_rho, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag]

            torch.save(organic_params, 'organic_params/' + self.run_name.replace('/','|') + '/' + 'weights.pt')
            fp = open('organic_params/' + self.run_name.replace('/','|') + '/' + 'hash.pt','w')
            fp.write(str(self.data_hash))
            fp.close()
        

        
        self.p_Psi.requires_grad=False
        self.p_rho.requires_grad=False
        self.WW_muq.requires_grad = False
        self.WW_inv_Sigmaq_diag.requires_grad=False
        self.BB_muq.requires_grad=False
        self.BB_inv_Sigmaq_diag.requires_grad=False


        print('Starting bandit training')
        
        #Necessary values
        
        dataloader_bandit = DataLoader(self.dataset_bandit, 
                                       batch_size=BS, 
                                       shuffle=True, 
                                       num_workers=6)
        K,P = self.K, self.P
        N = len(dataloader_bandit.dataset)
        
        
        #Psi for the location of the MN
        Psi_loc = self.p_Psi.detach().numpy()
        
        # define the Psi cov : 
        Psi_cov = Psi_loc
        if self.args['norm'] :
            Psi_cov /= np.linalg.norm(Psi_cov, axis = 0, keepdims=True)
        
        #Cholesky decomposition
        cov_k = Psi_cov.T.dot(Psi_cov)/self.P
        L = np.linalg.cholesky(cov_k)
        
       # tf.reset_default_graph()
        
        tf_psi_loc = tf.constant(Psi_loc)
        tf_psi_cov = tf.constant(Psi_cov)
        tf_L = tf.constant(L)

        #DEFINING PRIORS : 
        #Hyperparameters
        s_zeta = 1.0
        kappa_m = 0.0

        wa_m = self.args['wa_m']
        wb_m = self.args['wb_m']
        bias_m = self.args['wc_m']

        wa_s = self.args['wa_s']
        wb_s = self.args['wb_s']
        bias_s = self.args['wc_s']
        kappa_s = self.args['kappa_s']



        
        
        #W prior 0,1
        zeta_0_mean = tf.zeros([K**2, 1])
        zeta_0_std = s_zeta*tf.ones([K**2, 1])

        #Bias prior 
        bias_0_mean = tf.constant([bias_m])
        bias_0_std = tf.constant([bias_s])

        #W_a prior
        wa_0_mean = tf.constant([wa_m])
        wa_0_std = tf.constant([wa_s])

        #W_b prior
        wb_0_mean = tf.constant([wb_m])
        wb_0_std = tf.constant([wa_s])

        #Kappa prior 
        kappa_0_mean = tf.zeros([P,1])
        kappa_0_std = kappa_s*tf.ones([P,1])
        
        X = tf.placeholder(dtype=tf.float32, shape = [None, K])
        Y = tf.placeholder(dtype=tf.float32, shape = [None, 1])
        A = tf.placeholder(dtype=tf.int32, shape = [None,])

        #Zeta posterior
        zeta_means = tf.Variable(zeta_0_mean, dtype=tf.float32)
        zeta_logstd = tf.Variable(tf.log(zeta_0_std), dtype=tf.float32)

        #Bias posterior
        bias_means = tf.Variable(bias_0_mean, dtype=tf.float32)
        bias_logstd = tf.Variable(tf.log(bias_0_std), dtype=tf.float32)

        #wa posterior
        wa_means = tf.Variable(wa_0_mean, dtype=tf.float32)
        wa_logstd = tf.Variable(tf.log(wa_0_std), dtype=tf.float32)

        #wb posterior
        wb_means = tf.Variable(wb_0_mean, dtype=tf.float32)
        wb_logstd = tf.Variable(tf.log(wb_0_std), dtype=tf.float32)

        #Kappa posterior
        kappa_means = tf.Variable(kappa_0_mean, dtype=tf.float32)
        kappa_logstd = tf.Variable(tf.log(kappa_0_std), dtype=tf.float32)

        #Useful tensors :
        L_omega_u = tf.matmul(X, tf_L)
        Psi_a_u_loc = tf.gather(tf_psi_loc, A, axis = 0)
        Psi_a_u = tf.gather(tf_psi_cov, A, axis = 0)

        kappa_a_u = tf.gather(kappa_means, A, axis = 0)
        kappa_logstd_a_u = tf.gather(kappa_logstd, A, axis = 0)

        R = tf.reshape(tf.reshape(L_omega_u, [-1, 1, K])*tf.reshape(Psi_a_u, [-1, K, 1]), [-1, K**2])
        R_cov = tf.matmul(R**2, tf.exp(2*zeta_logstd))

        #ORGANIC :
        wa_noise = tf.random_normal([tf.shape(X)[0], 1])
        pred_org = sp(wa_means + tf.exp(wa_logstd)*wa_noise)*tf.reduce_sum(Psi_a_u_loc*X, axis = 1, keepdims=True) 

        #BANDIT :
        wb_noise = tf.random_normal([tf.shape(X)[0], 1])
        band_noise = tf.random_normal([tf.shape(X)[0], 1])
        pred_band =  sp(wb_means + tf.exp(wb_logstd)*wb_noise)*(tf.matmul(R, zeta_means) + tf.sqrt(R_cov)*band_noise)

        #Bias :
        bias_noise = tf.random_normal([tf.shape(X)[0], 1])
        pred_bias = bias_means + kappa_a_u + tf.sqrt(tf.exp(2*bias_logstd) + tf.exp(2*kappa_logstd_a_u))*bias_noise

        predictions = pred_org + pred_band + pred_bias

        pred_distribution = tfd.Bernoulli(logits=predictions)
        neg_log_prob = -tf.reduce_mean(pred_distribution.log_prob(Y))

        kl_zeta = KL_multivariate(zeta_means, zeta_logstd, zeta_0_mean, zeta_0_std)
        kl_bias = KL_multivariate(bias_means, bias_logstd, bias_0_mean, bias_0_std)
        kl_wa = KL_multivariate(wa_means, wa_logstd, wa_0_mean, wa_0_std)
        kl_wb = KL_multivariate(wb_means, wb_logstd, wb_0_mean, wb_0_std)
        kl_kappa = KL_multivariate(kappa_means, kappa_logstd, kappa_0_mean, kappa_0_std)

        kl_div =  kl_bias + kl_wa + kl_wb + kl_zeta + kl_kappa

        neg_ELBO = neg_log_prob + kl_div/N
        # Use ADAM optimizer w/ -ELBO loss
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(neg_ELBO)


        # Initialization op
        init_op = tf.global_variables_initializer()

        # Loop over the number of epochs
        
        # Run the training session
        with tf.Session() as sess:
            sess.run(init_op)
            for epoch in range(self.args['num_epochs_bandit']):
                # Training ------------------------------------------------------------------------------
                with tqdm(total=len(dataloader_bandit), desc='Epochs {}/{}'.format(epoch+1, self.args['num_epochs_bandit']), bar_format=bar_form) as pbar:
                    
                    for train_bandit in dataloader_bandit:
                        
                        sess_array, sess_mask, action, reward = train_bandit
                        sess_array, sess_mask = shrink_wrap(sess_array, sess_mask)
                        MB = sess_array.shape[0]
                        sess_mask = sess_mask.to(self.device)
                        sess_array = sess_array.to(self.device)

                        if self.use_em:
                            p_muq, p_inv_Sigmaq_diag, p_xi, p_a = VB_EM(sess_array.long(), sess_mask.float(), MB, self.P, self.K, self.p_Psi, 
                                                                        self.p_rho, self.device, self.cpu_device, Niter=100)
                        else:
                            p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array.long(), sess_mask.float(), MB, 
                                                                           self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, 
                                                                           self.BB_muq, self.BB_inv_Sigmaq_diag)

                        bx = p_muq.reshape(MB, self.K).detach().numpy()
                        by = reward.numpy().reshape(-1, 1)
                        ba = action.numpy()
                        
                        sess.run(train_op, feed_dict={X:bx, Y:by, A:ba})

                        pbar.update(1)

            print('bandit training complete')
            
            self.p_beta = torch.Tensor(sess.run(sp(wa_means)*tf_psi_loc + sp(wb_means)*tf.matmul(tf.matmul(tf_psi_cov, 
                                                                                                           tf.reshape(zeta_means, [K, K])), 
                                                                                                 tf.transpose(tf_L))))
            self.p_kappa = torch.Tensor(sess.run(kappa_means)[:,0])
            self.wa_lr = sess.run(wa_means)
            self.wb_lr = sess.run(wb_means)


    def create_user_embedding(self, sess_array, sess_mask):

        MB = sess_array.shape[0]
        sess_mask = sess_mask.to(self.device)
        sess_array = sess_array.to(self.device)
        p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array, sess_mask, MB, self.K, self.WW_muq, self.WW_inv_Sigmaq_diag, self.BB_muq, self.BB_inv_Sigmaq_diag)

        return p_muq, p_inv_Sigmaq_diag

    def bandit_prediction(self, sess_array, sess_mask):
        MB = sess_array.shape[0]
        sess_mask = sess_mask.to(self.device)
        sess_array = sess_array.to(self.device)
        if self.use_em:
            p_muq, p_inv_Sigmaq_diag, p_xi, p_a = VB_EM(sess_array, sess_mask, MB, self.P, self.K, self.p_Psi, 
                                                        self.p_rho, self.device, self.cpu_device, Niter=100)
        else:
            p_muq, p_inv_Sigmaq_diag = linear_auto_encoder(sess_array.long(), sess_mask, MB, self.K, 
                                                           self.WW_muq, self.WW_inv_Sigmaq_diag, 
                                                           self.BB_muq, self.BB_inv_Sigmaq_diag)
        user_embedding = p_muq.reshape(MB, self.K)
        logits = torch.matmul(user_embedding, self.p_beta.t()) + self.p_kappa
        return logits
