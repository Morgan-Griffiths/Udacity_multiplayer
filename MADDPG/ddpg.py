import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from noise import GaussianNoise,OUnoise
from models import Critic,Actor,hard_update,hidden_init
from Buffers.PER import PriorityReplayBuffer
from Buffers.priority_tree import PriorityTree
from Buffers.buffer import ReplayBuffer

class DDPG(object):
    def __init__(self,seed,nA,nS,config):
        self.seed = seed
        self.nA = nA
        self.nS = nS
        self.buffer_size = config.buffer_size
        self.min_buffer_size = config.min_buffer_size
        self.batch_size = config.batch_size
        self.L2 = config.L2
        self.tau = config.TAU
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.num_agents= config.num_agents
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.R = ReplayBuffer(nA,self.buffer_size,self.batch_size,seed)
        self.local_critic = Critic(seed,nS,nA).to(self.device)
        self.target_critic = Critic(seed,nS,nA).to(self.device)
        self.local_actor = Actor(seed,nS,nA).to(self.device)
        self.target_actor = Actor(seed,nS,nA).to(self.device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr = 1e-3,weight_decay=self.L2)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr = 1e-4)
        
        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)

    def step(self):
        pass

    def act(self):
        pass

    def learn(self):
        pass