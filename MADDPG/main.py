import numpy as np 
import gym
import time

from config import Config
from train import train
from unity_env import UnityEnv
from env_wrapper import MultiEnv
from maddpg import MultiAgent

"""
Instantiates config, multienv and multiagent
"""

def main(algo):
    K = 4
    ddpg_config = Config(algo)
    env = gym.make('MountainCarContinuous-v0')
    nS = env.observation_space.shape[0]
    nA = env.action_space.shape[0]
    K_envs = MultiEnv(env,nS,K)
    maddpg = MultiAgent(nS,nA,ddpg_config,K_envs,K)
    maddpg.train()


if __name__ == "__main__":
    algo = "maddpg"
    main(algo)