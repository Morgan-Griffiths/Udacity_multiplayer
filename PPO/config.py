"""
Config file for loading hyperparams
"""

import argparse

class Config(object):
    def __init__(self,agent):
        if agent == "PPO":
            self.gae_lambda=0.95
            self.num_agents=20
            self.batch_size=32
            self.gradient_clip=10
            self.SGD_epoch=10
            self.tmax = 320
            self.epsilon=0.2
            self.beta=0.01
            self.gamma=0.99
        elif agent == "DDPG":
            self.num_agents=20
            self.buffer_size = 10000
            self.min_buffer_size = 200
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
            self.UPDATE_EVERY = 4
            self.CLIP_NORM = 10
        else:
            raise ValueError('The Agent parameters of {} has not been implemented'.format(agent))