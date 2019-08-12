"""
Config file for loading hyperparams
"""

import argparse

class Config(object):
    def __init__(self,agent):
        if agent == "ddpg":
            self.seed = 99
            self.name = agent
            self.num_agents = 2
            self.QLR = 0.001
            self.ALR = 0.0001
            self.gamma = 0.99
            self.L2 = 0 # 0.1
            self.tau=0.01 # 0.001
            self.noise_decay=0.99995
            self.gae_lambda = 0.97
            self.CLIP_NORM = 10
            # Buffer
            self.buffer_size = int(1e6)
            self.min_buffer_size = int(1e3)
            self.batch_size = 256
            # Priority Replay
            self.ALPHA = 0.6 # 0.7 or 0.6
            self.START_BETA = 0.5 # from 0.5-1
            self.END_BETA = 1
            # distributional
            self.N_atoms = 51
            self.v_min = -2
            self.v_max = 2
            self.delta_z = (self.v_min - self.v_max) / (self.N_atoms - 1)
            #
            self.action_low=-1.0 
            self.action_high=1.0
            self.gamma=0.99
            self.update_every=1 
            self.update_repeat=1
            # Training
            self.episodes = 4000
            self.tmax = 2000
            self.print_every = 4
            self.SGD_epoch = 4
            self.checkpoint_path = 'model_weights/ddpg.ckpt'
            self.winning_condition = 0.5
        else:
            raise ValueError('Agent not implemented')