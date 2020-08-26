from recogym.agents import Agent
import torch
import pandas as pd
import numpy as np


class ModelBasedAgent(Agent):
    def __init__(self, config):
        super(ModelBasedAgent, self).__init__(config)

        self.data = {
            't': [],
            'u': [],
            'z': [],
            'v': [],
            'a': [],
            'c': [],
            'ps': [],
            'ps-a': [],
        }
        self.model = None

    def reset(self):
        self.history = [] 

    def act(self, observation, reward, done):
        """Make a recommendation"""

        # on the first action we train the model
        if self.model == None:            
            df = pd.DataFrame(self.data)
            df.to_csv('/tmp/datadump_3.csv')
            self.config.modelargs['dataset'] = df
            self.config.modelargs['P'] = self.config.num_products
            self.model = self.config.recomodel(self.config.modelargs)
            self.model.do_training()
            

        o = []
        for ii in range(len(observation.current_sessions)):
            o.append(observation.current_sessions[ii]['v'])
        sess_array = torch.Tensor(o).reshape(1,len(o))
        sess_mask = torch.ones_like(sess_array)
        if self.config.modelargs['organic']: # quite often sess_array seems to be empty here...  why is that?
            logits = self.model.next_item_prediction(sess_array, sess_mask)
        else:
            logits = self.model.bandit_prediction(sess_array, sess_mask)

        assert(logits.shape[1] == self.config.num_products)

        action = int(logits.argmax())
        ps_all = np.zeros(self.config.num_products)
        ps_all[action] = 1.0
        return {'a': action, 'ps': 1.0, 'ps-a': ps_all,'t': observation.context().time(), 'u': observation.context().user()}
        
    def train(self, observation, action, reward, done = False):
        if observation.sessions() == None:
            return
        for session in observation.sessions():
            self.data['t'].append(session['t'])
            self.data['u'].append(session['u'])
            self.data['z'].append('organic')
            self.data['v'].append(session['v'])
            self.data['a'].append(None)
            self.data['c'].append(None)
            self.data['ps'].append(None)
            self.data['ps-a'].append(None)

        if action != None:
            self.data['t'].append(action['t'])
            self.data['u'].append(action['u'])
            self.data['z'].append('bandit')
            self.data['v'].append(None)
            self.data['a'].append(action['a'])
            self.data['c'].append(reward)
            self.data['ps'].append(action['ps'])
            self.data['ps-a'].append(action['ps-a'])

