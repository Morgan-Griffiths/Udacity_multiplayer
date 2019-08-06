import copy
import numpy as np
import torch
import torch.nn.functional as F
import os

print(os.getcwd())
from Buffers.replay_buffer import ReplayBuffer
from Networks.models import Critic,Actor,hard_update
import torch.optim as optim

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

    def load_weights(self,path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.policy.state_dict(), path)

    def add(self,trajectory):
        self.R.add(trajectory)
        
    def step(self):
        # Sample memory if len > minimum
        # Get experience tuples
        samples = self.R.sample()
        # Learn from them and update local networks
        self.learn(samples)
        # Update target networks
        self.update_networks()
            
    def learn(self,samples):
        states,actions,rewards,next_states,dones = samples

        target_actions = self.target_actor(next_states)
        next_values = self.target_critic(next_states,target_actions).squeeze(-1)
        values = self.local_critic(states,actions)
    
        # GAE rewards
        gae_r,future_r = self.GAE_rewards(rewards,values,next_values,dones)
        # Reshape
        # next_values = next_values.view(self.batch_size*self.num_agents,1)
        # dones = dones.view(self.batch_size*self.num_agents,1)
        # states = states.view(self.batch_size*self.num_agents,1)
        # actions = actions.vie

        target_y = future_r + (self.gamma*next_values*(1-dones))
        current_y = self.local_critic(states,actions).squeeze(-1)

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss = (target_y - current_y).mean()**2
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(),1)
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer.zero_grad()
        local_actions = self.local_actor(states)
        actor_loss = -self.local_critic(states, local_actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
    def GAE_rewards(self,rewards,values,next_values,dones):
        # Return GAE and future rewards
        N = rewards.shape[0]
        A = self.num_agents
        # Get values and next_values in the same shape to quickly calculate TD_errors
        combined = self.gamma * self.gae_lambda

        # rewards = np.concatenate(rewards).reshape(N,A)
        next_values = np.concatenate(next_values.cpu().detach().numpy()).reshape(N,A)
        values = np.concatenate(values.cpu().detach().numpy()).reshape(N,A)
        TD_errors = rewards + next_values - values
        advs = np.zeros(rewards.shape)
        returns = np.zeros(rewards.shape)
        # For returns
        discounted_gamma = self.gamma**np.arange(N)
        j = 1
        for index in reversed(range(N)):
            P = N-index
            discounts = combined ** np.arange(0,N-index)
            returns[index,:] = np.sum(rewards[index:,:] * np.repeat([discounted_gamma[:j]],A,axis=1).reshape(P,A),axis=0)
            advs[index,:] = np.sum(TD_errors[index:,:] * np.repeat([discounts],A,axis=1).reshape(P,A),axis=0)
            j += 1
        # Normalize and reshape
        # returns = torch.from_numpy(returns.reshape(N*A,1)).float().to(self.device)
        # advs = torch.from_numpy(advs.reshape(N*A,1)).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        advs = torch.from_numpy(advs).float().to(self.device)
        advs = (advs - advs.mean()) / advs.std()
        return advs,returns
         
    def act(self,state,N):
        state = torch.from_numpy(state).float().to(self.device)
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).data.cpu().numpy()
        self.local_actor.train()
        # Act with noise
        action = np.clip(action + N,-1,1)
        return action
    
    def update_networks(self):
        self.target_critic = DDPG.soft_update_target(self.local_critic,self.target_critic,self.tau)
        self.target_actor = DDPG.soft_update_target(self.local_actor,self.target_actor,self.tau)
        
    @staticmethod
    def soft_update_target(local,target,tau):
        for local_param,target_param in zip(local.parameters(),target.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
        return target