
import torch
import torch.optim as optim
from collections import deque
import numpy as np

from models import Actor,Critic
from Buffers.buffer import ReplayBuffer
from noise import GaussianNoise,OUnoise
# from env_wrapper import MultiEnv
from models import hard_update

class MultiAgent(object):
    def __init__(self,nS,nA,config,K_envs,K):
        self.seed = config.seed
        self.nA = nA
        self.nS = nS
        self.episodes = config.episodes
        self.tmax = config.tmax
        self.print_every = config.print_every
        self.update_every = config.UPDATE_EVERY
        self.SGD_epoch = config.SGD_epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparams
        self.gamma = config.gamma
        self.buffer_size = config.buffer_size
        self.min_buffer_size = config.min_buffer_size
        self.batch_size = config.batch_size
        self.L2 = config.L2
        self.TAU = config.TAU
        
        # For multi agent
        self.K = K
        self.K_envs = K_envs
        self.R = ReplayBuffer

        # Instantiating Actor and Critic
        self.base_actor = Actor(self.seed,self.nS,self.nA)
        self.base_critic = Critic(self.seed,self.nS,self.nA)
        self.critic_optimizer = optim.Adam(self.base_critic.parameters(), lr = 1e-3,weight_decay=self.L2)
        self.actor_optimizer = optim.Adam(self.base_actor.parameters(), lr = 1e-4)

        # Instantiate the desired number of agents and envs
        self.local_critics = [Critic(self.seed,self.nS,self.nA).to(self.device) for agent in range(K)]
        self.local_actors = [Actor(self.seed,self.nS,self.nA).to(self.device) for agent in range(K)]
        self.target_critics = [Critic(self.seed,self.nS,self.nA).to(self.device) for agent in range(K)]
        self.target_actors = [Actor(self.seed,self.nS,self.nA).to(self.device) for agent in range(K)]
        
        # Copy the weights from base agents to target and local
        map(lambda x: hard_update(self.base_critic,x),self.local_critics)
        map(lambda x: hard_update(self.base_critic,x),self.target_critics)
        map(lambda x: hard_update(self.base_actor,x),self.local_actors)
        map(lambda x: hard_update(self.base_actor,x),self.target_actors)

    def load_weights(self,path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.policy.state_dict(), path)

    def act(self,states):
        # Remove backprop
        # map(lambda x: x.eval(), self.local_actors)
        # split states for each agent
        actions = [actor(state).detach().cpu().numpy() for actor,states in zip(self.local_actors,states)]
        # Set back to train
        # map(lambda x: x.train(), self.local_actors)
        actions = np.vstack(actions)
        return actions


    def learn(self):
        # update gradiants for critic and actor
        trajectories = self.R.sample()
        # update target networks
        self.update_targets()

    def update_targets(self):
        [maddpg.soft_update(critic,target) for critic,target in zip(self.local_critics,self.target_critics)]
        [maddpg.soft_update(actor,target) for actor,target in zip(self.local_actors,self.target_actors)]

    @staticmethod
    def soft_update(source,target):
        for param,target_param in zip(source.parameters(),target.parameters()):
            target_param.data.copy_(self.TAU * target_param.data + (1-self.Tau) * param.data)

    def train(self):
        """
        We stack and store the stacks as observations for critic training, 
        but keep the states and next states seperate for actor actions.
        """
        scores = []
        score_window = deque(maxlen=100)
        for e in range(1,self.episodes):
            episode_scores = []
            obs,states = self.K_envs.reset()
            for t in range(self.tmax):
                actions = self.act(states)
                next_obs,next_states,rewards,dones = self.K_envs.step(actions)

                # Store experience
                self.R.add(obs,actions,rewards,next_obs,dones)
                # Learn
                if t % self.update_every == 0 and len(self.R) > self.min_buffer_size:
                    for _ in range(self.SGD_epoch):
                        self.learn()
                states = next_states
                # Score tracking
                episode_scores.append(rewards)
            scores.append(np.mean(episode_scores))
            score_window.append(np.mean(episode_scores))
            if e % self.print_every == 0:
                print('Mean',score_window)
        