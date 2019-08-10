import numpy as np 
import gym
import time

from config import Config
from train import train
from unity_env import UnityEnv
from env_wrapper import MultiEnv
from maddpg import MultiAgent

"""
TODO 
- Separate the states and next_states so each actor acts only on its own.
- Don't need to have an env_wrapper, because its only two agents.
- rewards are generated separately for each agent.

Workflow:
load objects
    Env_wrapper
    MultiAgent
train
    act
    add experience
learn
sample experience
update gradients

To verify all algorithms work as intended

Verified so far:
Loading weights
Saving weights
Plotting works

Env_wrapper:
    reset
    step

MultiAgent Functions:
test (single agent acting)
act

Unverified:
Important:
MultiAgent functions:
- train
- state - action - next_state, order isn't messed up.
- Learning

test agent optimizers work 

"""

K = 2
ddpg_config = Config(algo)
env = gym.make('MountainCarContinuous-v0')
nS = env.observation_space.shape[0]
nA = env.action_space.shape[0]
K_envs = MultiEnv(env,nS,K)
maddpg = MultiAgent(nS,nA,ddpg_config,K_envs,K)

# 
