import copy
import numpy as np
import torch
import torch.nn.functional as F

from Buffers.priority_replay import PriorityReplayBuffer
from Networks.models import Critic,Actor,hard_update
import torch.optim as optim

class D4PG(object):
    def __init__(self,nA,nS,BUFFER_SIZE,min_buffer_size,batch_size,seed,L2,TAU,N_atoms,v_max,v_min,gamma=1.0,n_step=0.95):
        self.seed = seed
        self.nA = nA
        self.nS = nS
        self.Buffer_size = BUFFER_SIZE
        self.min_buffer_size = min_buffer_size
        self.batch_size = batch_size
        self.L2 = L2
        self.tau = TAU
        self.gamma = gamma
        self.n_step = n_step
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # For distributional
        self.N_atoms = N_atoms
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = v_min - v_max / (N_atoms - 1)
        self.atoms = np.linspace(v_min,v_max,N_atoms)
        
        self.R = PriorityReplayBuffer(nA,BUFFER_SIZE,batch_size,seed)
        self.local_critic = Critic(seed,nS,nA).to(self.device)
        self.target_critic = Critic(seed,nS,nA).to(self.device)
        self.local_actor = Actor(seed,nS,nA).to(self.device)
        self.target_actor = Actor(seed,nS,nA).to(self.device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr = 1e-3,weight_decay=L2)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr = 1e-4)
        
        # Copy the weights from local to target
        hard_update(self.local_critic,self.target_critic)
        hard_update(self.local_actor,self.target_actor)

    def add(self,state,action,reward,next_state,done):
        # Calculate TD_error
        
        # Add memory
        self.R.add(state,action,reward,next_state,done)

    def project_dist(self, target_z_dist, rewards, terminates):
        try:
        #next_distr = next_distr_v.data.cpu().numpy()

            rewards = rewards.reshape(-1)
            terminates = terminates.reshape(-1).astype(bool)
            #dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
            #batch_size = len(rewards)
            proj_distr = np.zeros((self.batch_size, self.n_atoms), dtype=np.float32)
            for atom in range(self.n_atoms):
                tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards + (self.v_min + atom * self.delta) * self.gamma))
                b_j = (tz_j - self.v_min) / self.delta
                l = np.floor(b_j).astype(np.int64)
                u = np.ceil(b_j).astype(np.int64)
                eq_mask = (u == l).astype(bool)
                proj_distr[eq_mask, l[eq_mask]] += target_z_dist[eq_mask, atom]
                ne_mask = (u != l).astype(bool)
                proj_distr[ne_mask, l[ne_mask]] += target_z_dist[ne_mask, atom] * (u - b_j)[ne_mask]
                proj_distr[ne_mask, u[ne_mask]] += target_z_dist[ne_mask, atom] * (b_j - l)[ne_mask]

            if terminates.any():
                proj_distr[terminates] = 0.0
                tz_j = np.minimum(self.v_max, np.maximum(self.v_min, rewards[terminates]))
                b_j = (tz_j - self.v_min) / self.delta
                l = np.floor(b_j).astype(np.int64)
                u = np.ceil(b_j).astype(np.int64)
                eq_mask = (u == l).astype(bool)
                eq_dones = terminates.copy()
                eq_dones[terminates] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l] = 1.0
                ne_mask = (u != l).astype(bool)
                ne_dones = terminates.copy()
                ne_dones[terminates] = ne_mask.astype(bool)
                if ne_dones.any():
                    proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
                    proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
        except Exception as e:
            print(e)
        return proj_distr
        
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
        Q_targets = self.target_critic(next_states,target_actions)
        target_y = rewards + (self.gamma*Q_targets*(1-dones))
    
        # GAE rewards
        # GAE_rewards = torch.tensor(self.GAE(rewards.cpu().numpy()))
        # target_y = GAE_rewards + (self.gamma*Q_targets*(1-dones))

        # update critic
        self.critic_optimizer.zero_grad()
        current_y = self.local_critic(states,actions)
        critic_loss = (target_y - current_y).mean()**2
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.local_critic.parameters(),1)
        self.critic_optimizer.step()

        # update actor
        self.actor_optimizer.zero_grad()
        local_actions = self.local_actor(states)
        actor_loss = -self.local_critic(states, local_actions)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        # if sum(rewards) > 0:
        #     print('finally')
        
    def GAE(self,rewards):
        """
        Generalized Advantage Estimate.
        N_step discounted returns
        """
        return np.sum([sum(rewards[:i+1])*((1-self.n_step)*self.n_step**i) for i in range(rewards.shape[0])])
         
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