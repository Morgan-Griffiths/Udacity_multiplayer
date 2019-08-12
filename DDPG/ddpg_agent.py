import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys


sys.path.append('/home/shuza/Code/Udacity_multiplayer')
from Buffers.replay_buffer import ReplayBuffer
from Buffers.PER import PriorityReplayBuffer
from Buffers.priority_tree import PriorityTree
from utils.noise import OUnoise
from models import hard_update

class Agent():
    def __init__(self, nS, nA, actor, critic,config):
        self.nS = nS
        self.nA = nA
        self.action_low = config.action_low
        self.action_high = config.action_high
        self.seed = config.seed

        self.tau = config.tau
        self.gamma = config.gamma
        self.update_every = config.update_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.L2 = config.L2
        self.update_repeat = config.update_repeat
        # noise
        self.noise = OUnoise(nA,config.seed)
        self.noise_scale = 1.0
        self.noise_decay = config.noise_decay

        # Priority Replay Buffer
        self.batch_size = config.batch_size
        self.buffer_size = config.buffer_size
        self.alpha = config.ALPHA
        self.beta = self.start_beta = config.START_BETA
        self.end_beta = config.END_BETA

        # actors networks
        self.actor = actor(self.seed,nS, nA).to(self.device)
        self.actor_target = actor(self.seed,nS, nA).to(self.device)

        # critic networks
        self.critic = critic(self.seed,nS, nA).to(self.device)
        self.critic_target = critic(self.seed,nS, nA).to(self.device)

        # Copy the weights from local to target
        hard_update(self.critic,self.critic_target)
        hard_update(self.actor,self.actor_target)

        # optimizer
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4, weight_decay=self.L2)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3, weight_decay=self.L2)

        # replay buffer
        self.PER = PriorityReplayBuffer(self.buffer_size, self.batch_size,self.seed,alpha=self.alpha,device=self.device)

        # reset agent for training
        self.reset_episode()
        self.it = 0

    def save_weights(self,path):
        params = {}
        params['actor'] = self.actor.state_dict()
        params['critic'] = self.critic.state_dict()
        torch.save(params, path)

    def load_weights(self,path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])

    def reset_episode(self):
        self.noise.reset()

    def act(self, state):
        with torch.no_grad():
            action = self.actor(self.tensor(state)).cpu().numpy()
        action += self.noise.sample() * self.noise_scale
        self.noise_scale = max(self.noise_scale * self.noise_decay, 0.01)
        self.actor.train()
        return np.clip(action, self.action_low, self.action_high)

    def evaluate(self,state):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(self.tensor(state)).cpu().numpy()
        return np.clip(action, self.action_low, self.action_high)

    def step(self, state, action, reward, next_state, done):

        next_action = self.actor(next_state)
        next_value = self.critic_target(next_state,next_action)
        target = reward + self.gamma * next_value * done
        local = self.critic(state,action)
        TD_error = target - local
        self.PER.add(state, action, reward, next_state, done, TD_error)
        
        self.it += 1
        if self.it < self.batch_size or self.it % self.update_every != 0:
            return
        for _ in range(self.update_repeat):
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.PER.sample()

        with torch.no_grad():
              target_actions = self.actor_target(next_states)
        next_values = self.critic_target(next_states,target_actions)
        y_target = rewards.unsqueeze(1) + self.gamma * next_values * (1-dones.unsqueeze(1))
        y_current = self.critic(states, actions)
        # update critic
        critic_loss = F.smooth_l1_loss(y_current, y_target)
        self.critic.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # update actor
        local_actions = self.actor(states)
        actor_loss = -self.critic(states, local_actions).mean()
        self.actor.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # soft update networks
        self.soft_update()

    def soft_update(self):
        """Soft update of target network
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    def tensor(self, x):
        return torch.from_numpy(x).float().to(self.device)
