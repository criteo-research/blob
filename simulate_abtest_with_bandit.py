from recogym import env_1_args

from models.models_organic_bandit import RecoModelRTAEWithBanditTF, RecoModelRTAEWithBanditTF_Full
from models.models_organic import RecoModelRTAE, RecoModelItemKNN
from models.liang_multivae import MultiVAE
from recogym.agents import organic_user_count_args
from recogym.agents import RandomAgent, random_args
from models.models_bandit import PyTorchMLRAgent, pytorch_mlr_args
from models.model_based_agents import ModelBasedAgent

from utils.utils_agents import eval_against_session_pop, first_element
import tensorflow as tf
import pandas as pd
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='rt/ae', help='algorithm path.')
    parser.add_argument('--batch_size', type=int, default= 1024, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.000001, help='Initial learning rate.')
    parser.add_argument('--K', type=int, default=10, help='K.')
    parser.add_argument('--rcatk', type=int, default=5, help='recall at')
    parser.add_argument('--P', type=int, default=20, help='Threshold products at this value Yoochoose.')
    parser.add_argument('--cudavis', type=str, default=None, help='restrict gpu usage')
    parser.add_argument('--units', type=list, default=None, help='ae units')
    parser.add_argument('--batch_size_test', type=int, default=5, help='batch size.')
    parser.add_argument('--reg_Psi', type=float, default=0.1, help='reg.')
    parser.add_argument('--reg_shift', type=float, default=0.0, help='reg.')
    parser.add_argument('--reg_rho', type=float, default=1, help='reg.')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to do test set eval on tb')
    parser.add_argument('--neg_samples', type=int, default=1, help='')


    parser.add_argument('--num_sessions', type=int, default= 100, help='')
    parser.add_argument('--num_sessions_organic', type=int, default=0, help='')
    parser.add_argument('--num_users_to_score', type=int, default= 100, help='')
    parser.add_argument('--organic_epochs', type=int, default=10, help='')
    parser.add_argument('--bandit_epochs', type=int, default=10, help='')


    args = parser.parse_args()
    dict_args = vars(args)
    num_sessions=args.num_sessions
    num_sessions_organic=args.num_sessions_organic

    sig_omega = 0.

    P = args.P
    dict_args['P'] = P

    r = []

    results_path = 'results/'
    num_users_to_score = args.num_users_to_score
    seed = 0
    latent_factor = 10
    num_organic_users_to_train = 0
    eval_fn = eval_against_session_pop
    log_eps = 0.3

    for rep in range(1, 20):
        for num_flips in [0, int(P/2)]:
            res = []

            # Organic LVM

            dict_args['lr'] = 0.0001
            dict_args['num_epochs'] = args.organic_epochs
            dict_args['use_em'] = False
            dict_args['organic'] = True
            dict_args['weight_path'] = 'weights/linAE'

            parameters = {
                'recomodel': RecoModelRTAE,
                'modelargs': dict_args,
                'num_products': P,
                'K': args.K
            }
            sc_lvm = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega, ModelBasedAgent, parameters,str(parameters['recomodel']), True)
            sc_lvm = first_element(sc_lvm,'LVM')
            res.append(sc_lvm)

            # LVM Bandit

            dict_args['lr'] = 0.001
            dict_args['num_epochs'] = args.organic_epochs
            dict_args['num_epochs_bandit'] = args.bandit_epochs
            dict_args['use_em'] = False
            dict_args['organic'] = False
            dict_args['norm'] = True
            dict_args['organic_weights'] = 'weights/linAE'

            dict_args['wa_m'] = -1.0
            dict_args['wb_m'] = -6.0
            dict_args['wc_m'] = -4.5

            dict_args['wa_s'] = 1.
            dict_args['wb_s'] = 1.
            dict_args['wc_s'] = 10.
            dict_args['kappa_s'] = 0.01

            parameters = {
                'recomodel': RecoModelRTAEWithBanditTF,
                'modelargs': dict_args,
                'num_products': P,
                'K': args.K,
            }

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                sc_bandit = eval_fn(P, num_sessions_organic, num_sessions,
                                            num_users_to_score, seed, latent_factor,
                                            num_flips, log_eps, sig_omega, ModelBasedAgent,
                                            parameters,str(parameters['recomodel']), True)

                sc_bandit = first_element(sc_bandit, 'LVM Bandit MVN-Q')
                res.append(sc_bandit)

            # LVM Bandit Gaussian instead of Matrix normal posterior

            dict_args['lr'] = 0.001
            dict_args['num_epochs'] = args.organic_epochs
            dict_args['num_epochs_bandit'] = args.bandit_epochs
            dict_args['use_em'] = False
            dict_args['organic'] = False
            dict_args['norm'] = True
            dict_args['organic_weights'] = 'weights/linAE'

            dict_args['wa_m'] = -1.0
            dict_args['wb_m'] = -6.0
            dict_args['wc_m'] = -4.5

            dict_args['wa_s'] = 1.
            dict_args['wb_s'] = 1.
            dict_args['wc_s'] = 10.
            dict_args['kappa_s'] = 0.01

            parameters = {
                'recomodel': RecoModelRTAEWithBanditTF_Full,
                'modelargs': dict_args,
                'num_products': P,
                'K': args.K,
            }

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                sc_bandit = eval_fn(P, num_sessions_organic, num_sessions,
                                            num_users_to_score, seed, latent_factor,
                                            num_flips, log_eps, sig_omega,  ModelBasedAgent,
                                            parameters,str(parameters['recomodel']), True)

                sc_bandit = first_element(sc_bandit, 'LVM Bandit NQ')
                res.append(sc_bandit)

            #CB

            pytorch_mlr_args['n_epochs'] = args.bandit_epochs
            pytorch_mlr_args['learning_rate'] =  0.01
            pytorch_mlr_args['ll_IPS'] =  False
            pytorch_mlr_args['alpha'] = 0.0
            pytorch_mlr_args['num_products'] = P
            sc_log_reg = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega, PyTorchMLRAgent, pytorch_mlr_args ,str(pytorch_mlr_args), True)
            sc_log_reg = first_element(sc_log_reg, 'CB')
            res.append(sc_log_reg)

            #log reg

            pytorch_mlr_args['n_epochs'] = args.bandit_epochs
            pytorch_mlr_args['learning_rate'] =  0.01
            pytorch_mlr_args['ll_IPS'] =  False
            pytorch_mlr_args['alpha'] = 1.0
            pytorch_mlr_args['num_products'] = P
            sc_log_reg = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips,  log_eps, sig_omega, PyTorchMLRAgent, pytorch_mlr_args ,str(pytorch_mlr_args), True)
            sc_log_reg = first_element(sc_log_reg, 'log reg')
            res.append(sc_log_reg)

            # Random

            random_args['P'] = P
            parameters = {
                        **organic_user_count_args,
                        **env_1_args,
                        'select_randomly': True,
                        'modelargs': random_args,
                    }

            sc_rand = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega,  RandomAgent, parameters,'randomagent', True)
            sc_rand = first_element(sc_rand,'random')
            res.append(sc_rand)


            # mean Itemknn

            dict_args['organic'] = True
            parameters = {
                'recomodel': RecoModelItemKNN,
                'modelargs': dict_args,
                'num_products': P,
                'K': 10}
            sc_itemknn = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips, log_eps, sig_omega, ModelBasedAgent, parameters ,str(parameters['recomodel']), True)
            sc_itemknn['model']='ItemKNN_mean'
            sc_itemknn = first_element(sc_itemknn,'ItemKNN_mean')
            res.append(sc_itemknn)



            # Liang multivae

            model_based_agent_args = {}
            model_based_agent_args['num_products'] = P
            model_based_agent_args['P'] = P

            model_based_agent_args['lam'] = 0.01
            model_based_agent_args['lr'] = 1e-3
            model_based_agent_args['random_seed'] = 98765
            model_based_agent_args['p_dims'] = [10, P]
            model_based_agent_args['q_dims']= None
            model_based_agent_args['num_epochs'] = args.organic_epochs
            model_based_agent_args['batch_size']= args.batch_size
            model_based_agent_args['organic'] = True

            # the total number of gradient updates for annealing
            model_based_agent_args['total_anneal_steps'] = 200000
            # largest annealing parameter
            model_based_agent_args['anneal_cap'] = 0.2

            lvm_parameters = {
                'recomodel': MultiVAE,
                'modelargs': model_based_agent_args,
                'num_products': P,
                'K': 10,
            }

            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                sc_mult = eval_fn(P, num_sessions_organic, num_sessions, num_users_to_score, seed, latent_factor, num_flips,  log_eps, sig_omega, ModelBasedAgent, lvm_parameters,str(lvm_parameters['recomodel']), True)
                sc_mult = first_element(sc_mult,'MultiVAE')
                res.append(sc_mult)

            results = pd.concat(res)
            results['seed']=seed
            results['flips']=num_flips
            results['logging'] = str(eval_fn).replace('<function ','').split(' at')[0].replace('eval_against_','')
            results['rep'] = str(rep)
            r.append(results)
            pd.concat(r).to_csv(results_path + 'interim_results.csv')
            print('saving')
            print(pd.concat(r))
            pd.concat(r).to_csv(results_path + 'interim_results_%d.csv' % rep)

    print(pd.concat(r))
    pd.concat(r).to_csv(results_path + 'final_results.csv')
