"""
Config file for loading hyperparams
"""

import argparse

class Config(object):
    def __init__(self,agent):
        if agent == "maddpg":
            self.seed = 1234
            self.name = agent
            self.num_agents=20
            self.buffer_size = 100000
            self.min_buffer_size = 1000
            self.batch_size = 25
            self.ALPHA = 0.6 # 0.7 or 0.6
            self.START_BETA = 0.5 # from 0.5-1
            self.END_BETA = 1
            self.QLR = 0.001
            self.ALR = 0.0001
            self.EPSILON = 1
            self.MIN_EPSILON = 0.01
            self.gamma = 0.995
            self.TAU = 0.001
            self.L2 = 0.01
            self.gae_lambda = 0.97
            self.CLIP_NORM = 10
            # distributional
            self.N_atoms = 51
            self.v_min = -2
            self.v_max = 2
            self.delta_z = (self.v_min - self.v_max) / (self.N_atoms - 1)
            # Training
            self.episodes = 50
            self.tmax = 200
            self.print_every = 4
            self.UPDATE_EVERY = 4
            self.SGD_epoch = 10
            self.actor_path = 'model_weights/actor/'
            self.critic_path = 'model_weights/critic/'
            self.winning_condition = 90
        else:
            raise ValueError('Agent not implemented')