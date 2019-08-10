
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import numpy as np
import time
import pickle

from models import Actor,Critic
from Buffers.buffer import ReplayBuffer
from noise import GaussianNoise,OUnoise
# from env_wrapper import MultiEnv
from models import hard_update
from plot import plot

class MultiAgent(object):
    def __init__(self,nS,nA,config,K_envs,K):
        self.seed = config.seed
        self.name = config.name
        self.nA = nA
        self.nS = nS
        self.episodes = config.episodes
        self.tmax = config.tmax
        self.print_every = config.print_every
        self.update_every = config.UPDATE_EVERY
        self.SGD_epoch = config.SGD_epoch
        self.actor_path = config.actor_path
        self.critic_path = config.critic_path
        # self.noise = GaussianNoise((K,nA),config.episodes)
        self.noise = OUnoise(K,config.seed)
        self.winning_condition = config.winning_condition
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparams
        self.gamma = config.gamma
        self.buffer_size = config.buffer_size
        self.min_buffer_size = config.min_buffer_size
        self.batch_size = config.batch_size
        self.L2 = config.L2
        self.tau = config.TAU
        
        # For multi agent
        self.K = K
        self.nO = K * nS # Observation space
        self.K_envs = K_envs
        self.R = ReplayBuffer(config.buffer_size,config.batch_size,config.seed)

        # Instantiating Actor and Critic
        self.base_actor = Actor(self.seed,self.nS,self.nA)
        self.base_critic = Critic(self.seed,self.nO,self.nA)

        # Instantiate the desired number of agents and envs
        self.local_critics = [Critic(self.seed,self.nO,self.nA).to(self.device) for agent in range(K)]
        self.local_actors = [Actor(self.seed,self.nS,self.nA).to(self.device) for agent in range(K)]
        self.target_critics = [Critic(self.seed,self.nO,self.nA).to(self.device) for agent in range(K)]
        self.target_actors = [Actor(self.seed,self.nS,self.nA).to(self.device) for agent in range(K)]

        # Instantiate optimizers
        self.critic_optimizers = [optim.Adam(self.local_critics[i].parameters(), lr = 1e-3,weight_decay=self.L2) for i in range(self.K)]
        self.actor_optimizers = [optim.Adam(self.local_actors[i].parameters(), lr = 1e-4) for i in range(self.K)]
        
        # Copy the weights from base agents to target and local
        map(lambda x: hard_update(self.base_critic,x),self.local_critics)
        map(lambda x: hard_update(self.base_critic,x),self.target_critics)
        map(lambda x: hard_update(self.base_actor,x),self.local_actors)
        map(lambda x: hard_update(self.base_actor,x),self.target_actors)

    def load_weights(self,critic_path,actor_path):
        # Load weigths from both

        # self.local_actors[0].load_state_dict(torch.load(actor_path+'0.ckpt'))
        # self.local_actors[0].eval()
        # self.local_actors[1].load_state_dict(torch.load(actor_path+'1.ckpt'))
        # self.local_actors[1].eval()
        [actor.load_state_dict(torch.load(actor_path+str(index)+'.ckpt')) for actor,index in zip(self.local_actors,range(self.K))]
        [actor.eval() for actor in self.local_actors]
        # self.local_critics = [critic.load_state_dict(torch.load(critic_path+str(index)+'.ckpt')) for critic,index in zip(self.local_critics,range(self.K))]
        

    def save_weights(self,critic_path,actor_path):
        # Save weights for both
        [torch.save(critic.state_dict(), critic_path+str(index)+'.ckpt') for critic,index in zip(self.local_critics,range(self.K))]
        [torch.save(actor.state_dict(), actor_path+str(index)+'.ckpt') for actor,index in zip(self.local_actors,range(self.K))]

    def train(self):
        """
        We stack and store the stacks as observations for critic training, 
        but keep the states and next states seperate for actor actions.
        """
        tic = time.time()
        means = []
        stds = []
        scores_window = deque(maxlen=100)
        for e in range(1,self.episodes):
            episode_scores = []
            obs,states = self.K_envs.reset()
            for t in range(self.tmax):
                actions = self.act(states)
                next_obs,next_states,rewards,dones = self.K_envs.step(actions)

                # Store experience
                self.R.add(obs,states,actions,rewards,next_obs,next_states,dones)
                # Learn
                if t % self.update_every == 0 and len(self.R) > self.min_buffer_size:
                    for _ in range(self.SGD_epoch):
                        # Update each agent
                        for i in range(self.K):
                            self.learn(i)
                states = next_states
                obs = next_obs
                # Score tracking
                episode_scores.append(np.sum(rewards))
                
            means.append(np.mean(episode_scores))
            stds.append(np.std(episode_scores))
            scores_window.append(np.mean(episode_scores))
            if e % 4 == 0:
                toc = time.time()
                r_mean = np.mean(scores_window)
                r_max = max(scores_window)
                r_min = min(scores_window)
                r_std = np.std(scores_window)
                plot(self.name,means,stds)
                print("\rEpisode: {} out of {}, Steps {}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,self.episodes,int(e*self.tmax*self.K),r_mean,r_min,r_max,r_std,(toc-tic)/60))
            if np.mean(scores_window) > self.winning_condition:
                print('Env solved!')
                # save scores
                pickle.dump([means,stds], open(str(self.name)+'_scores.p', 'wb'))
                # save policy
                self.save_weights(self.critic_path,self.actor_path)
                break
                
        
    def act(self,states):
        # split states for each agent
        actions = [actor(state).detach().cpu().numpy() for actor,state in zip(self.local_actors,states)]
        actions = np.vstack(actions)
        # Add noise for exploration
        actions = np.add(actions,self.noise.sample())
        return actions

    def test(self,state):
        # split states for each agent
        # actions = [actor(state).detach().cpu().numpy() for actor,state in zip(self.local_actors,states)]
        # actions = np.vstack(actions)
        action = self.local_actors[0](state).detach().cpu().numpy()
        return action

    def learn(self,index):
        # Get sample experiences
        obs,states,actions,rewards,next_obs,next_states,dones = self.R.sample()
        # Get target actions and target values
        self.critic_optimizers[index].zero_grad()
        with torch.no_grad():
            target_actions = self.target_actors[index](next_states)
        next_values = self.target_critics[index](next_obs,target_actions).detach()

        target_y = rewards + self.gamma * next_values * (1-dones)
        current_y = self.local_critics[index](obs,actions)

        critic_loss = F.smooth_l1_loss(current_y,target_y)
        critic_loss.backward()
        self.critic_optimizers[index].step()
        # Update actor
        self.actor_optimizers[index].zero_grad()
        local_actions = self.local_actors[index](states)
        actor_loss = -self.local_critics[index](obs,local_actions).mean()
        actor_loss.backward()
        self.actor_optimizers[index].step()
        # update target networks
        self.update_targets(index)

    def update_targets(self,index):
        MultiAgent.soft_update(self.local_critics[index],self.target_critics[index],self.tau)
        MultiAgent.soft_update(self.local_actors[index],self.target_actors[index],self.tau)

    def multi_update_targets(self):
        [MultiAgent.soft_update(critic,target,self.tau) for critic,target in zip(self.local_critics,self.target_critics)]
        [MultiAgent.soft_update(actor,target,self.tau) for actor,target in zip(self.local_actors,self.target_actors)]

    @staticmethod
    def soft_update(source,target,tau):
        for param,target_param in zip(source.parameters(),target.parameters()):
            target_param.data.copy_(tau * target_param.data + (1-tau) * param.data)