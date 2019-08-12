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
    def __init__(self,seed,nA,nS,L2,index):
        self.seed = seed
        self.nA = nA
        self.nS = nS
        self.nO = 52 # 24 * 2 state space + 2 * 2 action space 
        self.L2 = L2
        self.index = index
        self.noise = OUnoise(nA,seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.local_critic = Critic(seed,self.nO,nA).to(self.device)
        self.target_critic = Critic(seed,self.nO,nA).to(self.device)
        self.local_actor = Actor(seed,nS,nA).to(self.device)
        self.target_actor = Actor(seed,nS,nA).to(self.device)

        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)

        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr = 1e-3,weight_decay=self.L2)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr = 1e-4)
        
    def load_weights(self,critic_path,actor_path):
        # Load weigths from both
        self.local_critic.load_state_dict(torch.load(critic_path+'local_critic_'+str(self.index)+'.ckpt'))
        self.local_actor.load_state_dict(torch.load(actor_path+'local_actor_'+str(self.index)+'.ckpt'))
        self.target_critic.load_state_dict(torch.load(critic_path+'target_critic_'+str(self.index)+'.ckpt'))
        self.target_actor.load_state_dict(torch.load(actor_path+'target_actor_'+str(self.index)+'.ckpt'))
        self.local_actor.eval()
        
    def save_weights(self,critic_path,actor_path):
        # Save weights for both
        torch.save(self.local_actor.state_dict(), actor_path+'local_actor_'+str(self.index)+'.ckpt')
        torch.save(self.target_actor.state_dict(), actor_path+'target_actor_'+str(self.index)+'.ckpt')
        torch.save(self.local_critic.state_dict(), critic_path+'local_critic_'+str(self.index)+'.ckpt')
        torch.save(self.target_critic.state_dict(), critic_path+'target_critic_'+str(self.index)+'.ckpt')

    def act(self,state):
        action = self.local_actor(state).detach().cpu().numpy() + self.noise.sample()
        return action 

    def target_act(self,next_state):
        action = self.target_actor(next_state).detach().cpu().numpy() + self.noise.sample()
        return action

    def step(self):
        pass

    def learn(self):
        pass