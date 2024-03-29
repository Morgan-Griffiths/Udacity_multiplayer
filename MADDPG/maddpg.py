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
from ddpg import DDPG
from plot import plot
from utils import transpose_list,transpose_to_tensor

class MultiAgent(object):
    def __init__(self,env,nS,nA,config):
        self.seed = config.seed
        self.name = config.name
        self.nA = nA
        self.nS = nS
        self.num_agents = config.num_agents
        self.episodes = config.episodes
        self.tmax = config.tmax
        self.print_every = config.print_every
        self.update_every = config.UPDATE_EVERY
        self.SGD_epoch = config.SGD_epoch
        self.actor_path = config.actor_path
        self.critic_path = config.critic_path
        self.noise = GaussianNoise((self.num_agents,nA),config.episodes)
        # self.noise = OUnoise(nA,config.seed)
        self.winning_condition = config.winning_condition
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.device2 = 'cuda:1'
        else:
            self.device = 'cpu'

        # Hyperparams
        self.gamma = config.gamma
        self.buffer_size = config.buffer_size
        self.min_buffer_size = config.min_buffer_size
        self.batch_size = config.batch_size
        self.L2 = config.L2
        self.tau = config.TAU
        
        # For multi agent
        self.nO = self.num_agents * nS # Observation space
        self.env = env
        self.R = ReplayBuffer(config.buffer_size,config.batch_size,config.seed)

        # Instantiating Actor and Critic
        self.agents = [DDPG(self.seed,nA,nS,config.L2,0),DDPG(self.seed,nA,nS,config.L2,1)]

    def load_weights(self,critic_path,actor_path):
        # Load weigths from both
        [agent.load_weights(critic_path,actor_path) for agent in self.agents]
        
    def save_weights(self,critic_path,actor_path):
        # Save weights for both
        [agent.save_weights(critic_path,actor_path) for agent in self.agents]

    def seed_replay_buffer(self):
        obs = self.env.reset()
        while len(self.R) < self.min_buffer_size:
            for t in range(self.tmax):
                actions = ((np.random.rand(2,2)*2)-1)
                next_obs,rewards,dones = self.env.step(actions)
                # Store experience
                self.R.add(obs,actions,rewards,next_obs,dones)
                obs = next_obs

    def train(self):
        """
        We stack and store the stacks as observations for critic training, 
        but keep the states and next states seperate for actor actions.
        """
        tic = time.time()
        means = []
        stds = []
        steps = []
        scores_window = deque(maxlen=100)
        for e in range(1,self.episodes):

            self.noise.step()
            episode_scores = []
            net_hits = 0
            obs = self.env.reset()
            for t in range(self.tmax):
                actions = self.act(obs)
                next_obs,rewards,dones = self.env.step(actions)
                # Check rate of success
                if np.max(rewards) > 0:
                    net_hits += 1
                # Store experience
                self.R.add(obs,actions,rewards,next_obs,dones)
                # Score tracking
                if dones.any():
                    steps.append(int(t))
                episode_scores.append(np.max(rewards))
                obs = next_obs
            print('hit the ball over the net {} times'.format(net_hits))
            
            # Learn
            for _ in range(self.SGD_epoch):
                # Update each agent
                for i in range(self.num_agents):
                    self.learn(self.agents[i])
                # update target networks
                self.update_targets_all()
                
            means.append(np.mean(episode_scores))
            stds.append(np.std(episode_scores))
            scores_window.append(np.sum(episode_scores))
            if e % 4 == 0:
                toc = time.time()
                r_mean = np.mean(scores_window)
                r_max = max(scores_window)
                r_min = min(scores_window)
                r_std = np.std(scores_window)
                plot(self.name,means,stds)
                print("\rEpisode: {} out of {}, Steps {}, Mean steps {}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,self.episodes,np.sum(steps),np.mean(steps),r_mean,r_min,r_max,r_std,(toc-tic)/60))
            if np.mean(scores_window) > self.winning_condition:
                print('Env solved!')
                # save scores
                pickle.dump([means,stds], open(str(self.name)+'_scores.p', 'wb'))
                # save policy
                self.save_weights(self.critic_path,self.actor_path)
                break
        self.env.close()
                
        
    def act(self,obs):
        # split states for each agent
        actions = [agent.act(obs[i]) for i,agent in enumerate(self.agents)]
        actions = np.vstack(actions)
        return actions

    def evaluate(self,state):
        # TODO fix
        # Evaluate the agent's performance
        rewards = []
        
        obs = env.reset()
        for i in range(400):
            action = maddpg.act(obs)
            next_obs,reward,done = env.step(action)
            obs = next_obs
            rewards.append(np.sum(rewards))
            if done:
                break
        self.env.close()
        print("The agent achieved an average score of {:.2f}".format(np.mean(rewards)))
        return action

    def target_act(self,next_states):
        target_actions = torch.from_numpy(np.vstack([agent.target_act(next_states[:,index,:].reshape(self.batch_size,1,24)) for index,agent in enumerate(self.agents)]).reshape(self.batch_size,2,2)).float().to(self.device)
        return target_actions

    def local_act(self,states):
        local_actions = torch.from_numpy(np.vstack([agent.act(states[:,index,:].reshape(self.batch_size,1,24)) for index,agent in enumerate(self.agents)]).reshape(self.batch_size,2,2)).float().to(self.device)
        return local_actions

    def learn(self,agent):
        # Get sample experiences
        obs,actions,rewards,next_obs,dones = self.R.sample()
        # Get target actions and target values
        agent.critic_optimizer.zero_grad()
        target_actions = self.target_act(next_obs)
        # stack actions and observations for single critic input
        target_critic_input = torch.cat((next_obs,target_actions),dim=-1).view(self.batch_size,52)
        with torch.no_grad():
            next_values = agent.target_critic(target_critic_input).detach().squeeze(1)

        target_y = rewards[:,agent.index] + self.gamma * next_values * (1-dones[:,agent.index])
        # stack actions and observations for single critic input
        local_critic_input = torch.cat((obs,actions),dim=-1).view(self.batch_size,52)
        current_y = agent.local_critic(local_critic_input)

        critic_loss = F.smooth_l1_loss(current_y,target_y)
        critic_loss.backward()
        agent.critic_optimizer.step()
        # Update actor
        agent.actor_optimizer.zero_grad()
        local_actions = self.local_act(obs)
        actor_critic_input = torch.cat((obs,local_actions),dim=-1).view(self.batch_size,52)
        actor_loss = -agent.local_critic(actor_critic_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

    def update_targets(self,index):
        MultiAgent.soft_update(self.local_critics[index],self.target_critics[index],self.tau)
        MultiAgent.soft_update(self.local_actors[index],self.target_actors[index],self.tau)

    def update_targets_all(self):
        for agent in self.agents:
            MultiAgent.soft_update(agent.local_critic,agent.target_critic,self.tau)
            MultiAgent.soft_update(agent.local_actor,agent.target_actor,self.tau)

    def multi_update_targets(self):
        [MultiAgent.soft_update(critic,target,self.tau) for critic,target in zip(self.local_critics,self.target_critics)]
        [MultiAgent.soft_update(actor,target,self.tau) for actor,target in zip(self.local_actors,self.target_actors)]

    @staticmethod
    def soft_update(source,target,tau):
        for param,target_param in zip(source.parameters(),target.parameters()):
            target_param.data.copy_(tau * target_param.data + (1-tau) * param.data)