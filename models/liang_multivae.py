import seaborn as sn
sn.set()

from pandas.util import hash_pandas_object
import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import *




# from https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb

class MultiDAE(object):
    def __init__(self, p_dims, q_dims=None, lam=0.01, lr=1e-3, random_seed=None):
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        self.dims = self.q_dims + self.p_dims[1:]
        
        self.lam = lam
        self.lr = lr
        self.random_seed = random_seed

        self.construct_placeholders()

    def construct_placeholders(self):        
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dims[0]])
        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)

    def build_graph(self):

        self.construct_weights()

        saver, logits = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        # per-user average negative log-likelihood
        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph, axis=1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        reg_var = apply_regularization(reg, self.weights)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        loss = neg_ll + 2 * reg_var
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('loss', loss)
        merged = tf.summary.merge_all()
        return saver, logits, loss, train_op, merged

    def forward_pass(self):
        # construct forward graph        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights) - 1:
                h = tf.nn.tanh(h)
        return tf.train.Saver(), h

    def construct_weights(self):

        self.weights = []
        self.biases = []
        
        # define weights
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            weight_key = "weight_{}to{}".format(i, i+1)
            bias_key = "bias_{}".format(i+1)
            
            self.weights.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights[-1])
            tf.summary.histogram(bias_key, self.biases[-1])

class MultiVAE(MultiDAE):
    def __init__(self, args):
        self.cpu_device = torch.device('cpu')        
        self.device = self.cpu_device

        self.p_dims = args['p_dims']
        if args['q_dims'] is None:
            self.q_dims = args['p_dims'][::-1]
        else:
            assert args['q_dims'][0] == args['p_dims'][-1], "Input and output dimension must equal each other for autoencoders."
            assert args['q_dims'][-1] == args['p_dims'][0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = args['q_dims']
        self.dims = self.q_dims + self.p_dims[1:]
        
        self.lam = args['lam']
        self.lr = args['lr']
        self.random_seed = args['random_seed']
        self.P = args['num_products']
        self.args = args

        self.construct_placeholders()

        # Dataset stuff
        if isinstance(args['dataset'], str):
            fn_train = args['dataset'] + '/train.csv'
            fn_test = args['dataset'] + '/test.csv'
            self.dataset = PandasDataset(args['P'], pd.read_csv(fn_train), testing=True)
            self.dataset_test = PandasDataset(args['P'], pd.read_csv(fn_test), testing=True)

            self.data_hash = None

        else: # support the direct insertion of a dataframe into the dataset
            self.dataset = PandasDataset(args['P'], args['dataset'], testing=True)

            self.data_hash = hash_pandas_object(args['dataset'].v).sum()

    def construct_placeholders(self):
        super(MultiVAE, self).construct_placeholders()

        # placeholders with default values when scoring
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)
        
    def build_graph(self):
        self._construct_weights()

        saver, logits, KL = self.forward_pass()
        log_softmax_var = tf.nn.log_softmax(logits)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * self.input_ph,
            axis=-1))
        # apply regularization to weights
        reg = l2_regularizer(self.lam)
        
        reg_var = apply_regularization(reg, self.weights_q + self.weights_p)
        # tensorflow l2 regularization multiply 0.5 to the l2 norm
        # multiply 2 so that it is back in the same scale
        neg_ELBO = neg_ll + self.anneal_ph * KL + 2 * reg_var
        
        train_op = tf.train.AdamOptimizer(self.lr).minimize(neg_ELBO)

        # add summary statistics
        tf.summary.scalar('negative_multi_ll', neg_ll)
        tf.summary.scalar('KL', KL)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO)
        merged = tf.summary.merge_all()

        return saver, logits, neg_ELBO, train_op, merged
    
    def q_graph(self):
        mu_q, std_q, KL = None, None, None
        
        h = tf.nn.l2_normalize(self.input_ph, 1)
        h = tf.nn.dropout(h, self.keep_prob_ph)
        
        for i, (w, b) in enumerate(zip(self.weights_q, self.biases_q)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_q) - 1:
                h = tf.nn.tanh(h)
            else:
                mu_q = h[:, :self.q_dims[-1]]
                logvar_q = h[:, self.q_dims[-1]:]

                std_q = tf.exp(0.5 * logvar_q)
                KL = tf.reduce_mean(tf.reduce_sum(
                        0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q**2 - 1), axis=1))
        return mu_q, std_q, KL

    def p_graph(self, z):
        h = z
        
        for i, (w, b) in enumerate(zip(self.weights_p, self.biases_p)):
            h = tf.matmul(h, w) + b
            
            if i != len(self.weights_p) - 1:
                h = tf.nn.tanh(h)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))

        sampled_z = mu_q + self.is_training_ph *\
            epsilon * std_q

        # p-network
        logits = self.p_graph(sampled_z)
        
        return tf.train.Saver(), logits, KL

    def _construct_weights(self):
        self.weights_q, self.biases_q = [], []
        
        for i, (d_in, d_out) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims[:-1]) - 1:
                # we need two sets of parameters for mean and variance,
                # respectively
                d_out *= 2
            weight_key = "weight_q_{}to{}".format(i, i+1)
            bias_key = "bias_q_{}".format(i+1)
            
            self.weights_q.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_q.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_q[-1])
            tf.summary.histogram(bias_key, self.biases_q[-1])
            
        self.weights_p, self.biases_p = [], []

        for i, (d_in, d_out) in enumerate(zip(self.p_dims[:-1], self.p_dims[1:])):
            weight_key = "weight_p_{}to{}".format(i, i+1)
            bias_key = "bias_p_{}".format(i+1)
            self.weights_p.append(tf.get_variable(
                name=weight_key, shape=[d_in, d_out],
                initializer=tf.contrib.layers.xavier_initializer(
                    seed=self.random_seed)))
            
            self.biases_p.append(tf.get_variable(
                name=bias_key, shape=[d_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.001, seed=self.random_seed)))
            
            # add summary stats
            tf.summary.histogram(weight_key, self.weights_p[-1])
            tf.summary.histogram(bias_key, self.biases_p[-1])

    def do_training(self):
        BS = self.args['batch_size']
        dataloader = DataLoader(self.dataset, batch_size=BS, shuffle=False, num_workers=6)


        saver, logits_var, loss_var, train_op_var, merged_var = self.build_graph()


        # the total number of gradient updates for annealing
        total_anneal_steps = self.args['total_anneal_steps']
        # largest annealing parameter
        anneal_cap = self.args['anneal_cap']



        ndcgs_vad = []
        bar_form = '{l_bar}{bar} | {n_fmt}/{total_fmt} [{remaining} {postfix}]'

        bnum = 0 # what is the difference between this and update_count?  delete

        with tf.Session() as sess:

            init = tf.global_variables_initializer()
            sess.run(init)

            update_count = 0.0

            for epoch in range(self.args['num_epochs']):

                # Training ------------------------------------------------------------------------------
#                 with tqdm(total=len(dataloader), desc='Epochs {}/{}'.format(epoch+1, self.args['num_epochs']), bar_format=bar_form) as pbar:

                    for train in tqdm(dataloader):

                        sess_array, sess_mask, sessids, lastv = train
                        lastv = lastv.to(self.device)
                        sess_array, sess_mask = shrink_wrap(sess_array, sess_mask)
                        MB = sess_array.shape[0]
                        sess_mask = sess_mask.to(self.device)
                        sess_array = sess_array.to(self.device)

                        X = to_count_rep(sess_array, sess_mask, MB, self.P, self.device).detach().numpy().reshape(MB,self.P)

                        X = X.astype('float32') 
                        
                        if total_anneal_steps > 0:
                            anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                        else:
                            anneal = anneal_cap
                        
                        feed_dict = {self.input_ph: X, 
                                        self.keep_prob_ph: 0.5, 
                                        self.anneal_ph: anneal,
                                        self.is_training_ph: 1}        
                        sess.run(train_op_var, feed_dict=feed_dict)

                        if False:
                            if bnum % 100 == 0:
                                summary_train = sess.run(merged_var, feed_dict=feed_dict)
                                summary_writer.add_summary(summary_train, 
                                                            global_step=epoch * batches_per_epoch + bnum) 
                        
                        update_count += 1
                        bnum += 1
                        
            self.weights_p_np =  [(sess.run(wp), sess.run(bp)) for (wp, bp) in zip(self.weights_p, self.biases_p)] # x|z
            self.weights_q_np =  [(sess.run(wq), sess.run(bq)) for (wq, bq) in zip(self.weights_q, self.biases_q)] # z|x
            
            if 'weight_path_tf' in self.args:
                print('saving weights to ' + self.args['weight_path_tf'])
                fp = open(self.args['weight_path_tf'] + '_hash','w')
                fp.write(str(self.data_hash))
                fp.close()

                fp = open(self.args['weight_path_tf'],'wb')
                pickle.dump([self.weights_p_np, self.weights_q_np], fp)
                fp.close()


            
    def next_item_prediction(self, sess_array, sess_mask):
        
        MB = sess_array.shape[0]
        X = to_count_rep(sess_array, sess_mask, MB, self.P, self.device).detach().numpy().reshape(MB,self.P)
        X = X.astype('float32')
        h = X/(np.linalg.norm(X, axis=-1) + 1e-8)

        for w, b in self.weights_q_np :
            h = np.tanh(h.dot(w) + b) # Otmane does this look ok?
        logits = h[:, :self.q_dims[-1]]
        for w,b in self.weights_p_np :
            logits = logits.dot(w) + b
            
        return logits