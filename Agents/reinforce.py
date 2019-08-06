import numpy as np
import torch
from collections import deque

from Networks.models import Policy 

class REINFORCE(object):
    def __init__(self,env,nA,nS,seed,episodes=1000,tmax = 700,discount = 0.995, epsilon=0.1, beta=0.01,gamma=1.0,n_step=0.95):
        self.seed = seed
        self.env = env
        self.nA = nA
        self.nS = nS
        self.episodes = episodes
        self.tmax = tmax

        self.gamma = gamma
        self.n_step = n_step
        self.discount = discount
        self.start_epsilon = self.epsilon = epsilon
        self.start_beta = self.beta = beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Policy(seed,nS,nA)

    def learn(self,state):
        pass

    def step(self):
        pass

    def gather_trajectories(self):
        pass

    def train(self):
        scores_window = deque(maxlen=100)
        scores = []
        for e in range(self.episodes):
            score = 0
            state = env.reset()
            for i in range(self.tmax):
                action = self.policy(state)
                next_state, reward = env.step(action)
                loss = -m.log_prob(action) * reward
                loss.backward()
                state = next_state
                score += reward

            scores_window.append(score)
            print('\rEpisode {}\t Average Score: {:.2f}'.format(e, np.mean(scores_window)),end="")
        
        # if np.mean(scores_window) >= 200.0:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e, np.mean(scores_window)))

