import numpy as np
import torch
import random
from collections import namedtuple,deque
"""
Multi Agent buffer
"""
class ReplayBuffer(object):
    def __init__(self,buffer_size,batch_size,seed):
        
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.experience = namedtuple('experience',field_names=['obs','state','actions','rewards','next_obs','next_states','dones'])
        self.memory = deque(maxlen=buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('buffer device',self.device)
    
    def sample(self):
        experiences = random.sample(self.memory,self.batch_size)
        
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        # # convert dones to uint from bools
        # dones = torch.from_numpy(np.vstack([np.array(e.done,dtype=np.uint8) for e in experiences if e is not None])).float().to(self.device)
        obs = torch.from_numpy(np.vstack([e.obs for e in experiences if e is not None])).float()#.to(self.device)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()#.to(self.device)
        actions = torch.from_numpy(np.vstack([e.actions for e in experiences if e is not None])).float()#.to(self.device)
        rewards = torch.from_numpy(np.vstack([e.rewards for e in experiences if e is not None])).float()#.to(self.device)
        next_obs = torch.from_numpy(np.vstack([e.next_obs for e in experiences if e is not None])).float()#.to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_states for e in experiences if e is not None])).float()#.to(self.device)
        dones = torch.from_numpy(np.vstack([np.array(e.dones,dtype=np.uint8) for e in experiences if e is not None])).float()#.to(self.device)
        
        return (obs,states,actions,rewards,next_obs,next_states,dones)
    
    def add(self,obs,states,actions,rewards,next_obs,next_states,dones):
        for i in range(obs.shape[0]):
            e = self.experience(obs,states[i],actions[i],rewards[i],next_obs,next_states[i],dones[i])
            self.memory.append(e) 
    
    def __len__(self):
        return len(self.memory)