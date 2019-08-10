import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import os

# from noise import initialize_N
from Networks.models import Policy
from unity_env import UnityEnv
import torch.optim as optim

class PPO(object):
    def __init__(self,nA,nS,seed,device,config):
        self.seed = seed
        self.nA = nA
        self.nS = nS
        self.gae_lambda = config.gae_lambda
        self.batch_size= config.batch_size
        self.tmax = config.tmax
        self.start_epsilon = self.epsilon = config.epsilon
        self.start_beta = self.beta = config.beta
        self.gamma = config.gamma
        self.gradient_clip = config.gradient_clip
        self.SGD_epoch = config.SGD_epoch
        self.device = device

        self.policy = Policy(self.device,seed,nS,nA).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = 1e-4)
        # self.optimizer = AdamHD(self.policy.parameters(), lr = 1e-4)

    def load_weights(self,path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()

    def save_weights(self,path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        torch.save(self.policy.state_dict(), path)

    def reset_hyperparams(self):
        self.discount = self.start_discount
        self.epsilon = self.start_epsilon
        self.beta = self.start_beta

    def step_hyperparams(self):
        self.epsilon *= 0.999
        self.beta *= 0.995
    
    def step(self,trajectory):
        states,next_states,actions,log_probs,dones,values,rewards = trajectory
        advantages,rewards,returns = self.return_advs(values,dones,next_states[-1],rewards)
        self.learn_good(states,next_states,actions,log_probs,dones,returns,advantages)

    def act(self,state):
        with torch.no_grad():
            action,log_prob,dist,value = self.policy(state)
        action,log_prob,value = action.cpu().numpy(), log_prob.cpu().numpy(),value.cpu().squeeze(-1)
        return action,log_prob,dist,value
        
    def return_advs(self,values,dones,last_state,rewards):
        N = len(values)
        self.policy.eval()
        # Get values and next_values in the same shape to quickly calculate TD_errors
        with torch.no_grad():
            next_value = self.policy(last_state)[-1].cpu().squeeze(-1)
        self.policy.train()
        next_values = np.array(values[1:] + [next_value])
        values = np.array(values)
        combined = self.gamma * self.gae_lambda
        TD_errors = rewards + next_values - values
        advs = np.zeros(rewards.shape)
        # For returns
        discounted_gamma = self.gamma**np.arange(N)
        discounted_returns = rewards*discounted_gamma
        returns = discounted_returns.cumsum()[::-1] * dones
        j = 1
        for index in reversed(range(N)):
            P = N-index
            discounts = combined ** np.arange(0,N-index)
            advs[index] = np.sum(TD_errors[index:] * discounts)
            j += 1
        # Normalize and reshape
        advs = advs * dones
        returns = torch.from_numpy(returns).float().to(self.device)
        advs = torch.from_numpy(advs).float().to(self.device)
        advs = (advs - advs.mean()) / advs.std()
        return advs,rewards,returns

    def learn_good(self,states,next_states,actions,log_probs,dones,returns,advantages):
        # reshape so that memory is shared
        N = len(states)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(self.device)
        log_probs = torch.from_numpy(np.vstack(log_probs)).float().to(self.device)
        # values = torch.from_numpy((values).reshape(N,1)).float().to(self.device)
        # do multiple training runs on the data
        for _ in range(self.SGD_epoch):
            # Iterate through random batch generator
            for start in self.minibatch(N):
                # For training on sequences
                end = min(start+self.batch_size,N)
                states_b = states[start:end]
                actions_b = actions[start:end]
                log_probs_b = log_probs[start:end]
                returns_b = returns[start:end]
                advantages_b = advantages[start:end]
                # for training on random batches (minibatches must be modified)
                # states_b = states[indicies]
                # actions_b = actions[indicies]
                # log_probs_b = log_probs[indicies]
                # values_b = values[indicies]
                # returns_b = returns[indicies]
                # advantages_b = advantages[indicies]

                # get new probabilities with grad to perform the update step
                # Calculate the ratio between old and new log probs. Get loss and update grads
                _,new_log_probs,entropy,new_values = self.policy(states_b,actions_b)

                ratio = (new_log_probs - log_probs_b).exp().mean(dim=1)
                # ratio = new_log_probs / log_probs_b

                clip = torch.clamp(ratio,1-self.epsilon,1+self.epsilon)
                clipped_surrogate = torch.min(ratio*advantages_b, clip*advantages_b)


                actor_loss = -torch.mean(clipped_surrogate) - self.beta * entropy.mean()
                critic_loss = F.smooth_l1_loss(returns_b,new_values.squeeze(-1))
                
                self.optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.gradient_clip)
                self.optimizer.step()
                self.step_hyperparams()

    def minibatch(self,N):
        indicies = np.arange(min(self.batch_size,N))
        for _ in range(self.SGD_epoch):
            yield np.random.choice(indicies)