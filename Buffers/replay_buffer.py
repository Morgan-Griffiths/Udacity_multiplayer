import numpy as np
import torch
import random
from collections import namedtuple,deque

class ReplayBuffer(object):
    def __init__(self,action_size,buffer_size,batch_size,seed):
        
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple('experience',field_names=['state','action','reward','next_state','done'])
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
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([np.array(e.done,dtype=np.uint8) for e in experiences if e is not None])).float().to(self.device)
        
        return (states,actions,rewards,next_states,dones)
    
    def add(self,trajectory):
        state,action,reward,next_state,done = trajectory
#         print('action',action)
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e) 
    
    def __len__(self):
        return len(self.memory)