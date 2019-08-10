import numpy as np 
import time
import sys
import os

from config import Config
from train import train
from unity_env import UnityEnv
from env_wrapper import MultiEnv
from maddpg import MultiAgent

"""
Instantiates config, multienv and multiagent
"""
def main(algo):
    seed = 7

    # Load the ENV
    ### For running in VSCode ###
    # env = UnityEnv(env_file='Environments/Tennis_Linux/Tennis.x86_64',no_graphics=True)
    ### For running from terminal ###
    env = UnityEnv(env_file='../Environments/Tennis_Linux/Tennis.x86_64',no_graphics=True)

    # number of agents
    num_agents = env.num_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = env.action_size

    # examine the state space 
    state_size = env.state_size
    print('Size of each action: {}, Size of the state space {}'.format(action_size,state_size))
    
    ddpg_config = Config(algo)

    maddpg = MultiAgent(env,state_size,action_size,ddpg_config)
    maddpg.train()


if __name__ == "__main__":
    algo = "maddpg"
    main(algo)