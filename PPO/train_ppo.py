import numpy as np
from collections import deque
import time
import pickle
import torch

from plot import plot

def train_ppo(env,agent,episodes,path):
    name = 'PPO'
    tic = time.time()
    means = []
    stds = []
    steps = 0
    rewards_sum = deque(maxlen=100)
    for i_episode in range(1,episodes):#,episodes+1):
        # get trajectories
        trajectories = []
        while len(trajectories) < 300:
            trajectory_a,trajectory_b,rewards = collect_trajectories(env,agent)
            trajectories.append(trajectory_a + trajectory_b)
            
            rewards_sum.append(np.sum(rewards))
            means.append(np.mean(rewards_sum))
            stds.append(np.std(rewards_sum))

        agent.step(trajectories)
        steps += len(trajectories[0][0])
        # scores = np.sum(np.concatenate(episode_score).reshape(len(states),self.num_agents),axis=1)
        # get the average reward of the parallel environments

        if i_episode % 10 == 0:
            toc = time.time()
            r_mean = np.mean(rewards_sum)
            r_max = max(rewards_sum)
            r_min = min(rewards_sum)
            r_std = np.std(rewards_sum)
            plot(name,means,stds)
            print("\rEpisode: {} out of {}, Steps {}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(i_episode,episodes,steps,r_mean,r_min,r_max,r_std,(toc-tic)/60))
            if r_mean > 0.5:
                print('Env solved!')
                # save scores
                pickle.dump([means,stds], open('ppo_scores.p', 'wb'))
                # save policy
                agent.save_weights(path)
                break

def collect_trajectories(env,agent):
        """
        Playing against itself, the states,next_states,rewards must be separated for each agent. the actions must be for both agents. 
        I can train on only local observations, with two forward passes, one for each agent.
        I need to separate the (state,action,reward,next_state) tuples for the respective agents.
        """
        states_a,next_states_a,actions_a,log_probs_a,dones_a,values_a = [],[],[],[],[],[]
        states_b,next_states_b,actions_b,log_probs_b,dones_b,values_b = [],[],[],[],[],[]
        rewards_a = []
        rewards_b = []
        total_rewards = []
        state = env.reset()
        state_a = state[0,:]
        state_b = state[1,:]
        for t in range(200):
            with torch.no_grad():
                action_a,log_prob_a,dist_a,value_a = agent.act(state_a)
                action_b,log_prob_b,dist_b,value_b = agent.act(state_b)
            action = np.vstack((action_a,action_b))
            next_state,reward,done = env.step(action)
            next_state_a = next_state[0,:]
            next_state_b = next_state[1,:]
            reward_a = reward[0]
            reward_b = reward[1]
            # For ease of multiplication later
            inverse_done = np.logical_not(done).astype(int)
            inverse_dones_a = inverse_done[0]
            inverse_dones_b = inverse_done[1]

            rewards_a.append(reward_a)
            rewards_b.append(reward_b)
            states_a.append(np.expand_dims(state_a,axis=0))
            states_b.append(np.expand_dims(state_b,axis=0))
            next_states_a.append(next_state_a)
            next_states_b.append(next_state_b)
            actions_a.append(action_a)
            actions_b.append(action_b)
            log_probs_a.append(log_prob_a)
            log_probs_b.append(log_prob_b)
            dones_a.append(inverse_dones_a)
            dones_b.append(inverse_dones_b)
            values_a.append(value_a.numpy())
            values_b.append(value_b.numpy())
            state_a = next_state_a
            state_b = next_state_b
            if done.any():
                break
        episode_score = max(np.sum(rewards_a),np.sum(reward_b))
        trajectory_a = (states_a,next_states_a,actions_a,log_probs_a,np.array(dones_a),values_a,np.array(rewards_a))
        trajectory_b = (states_b,next_states_b,actions_b,log_probs_b,np.array(dones_b),values_b,np.array(rewards_b))
        return trajectory_a,trajectory_b,episode_score

def collect_trajectories_split(env,agent_a,agent_b):
        """
        Playing against itself, the states,next_states,rewards must be separated for each agent. the actions must be for both agents. 
        I can train on only local observations, with two forward passes, one for each agent.
        I need to separate the (state,action,reward,next_state) tuples for the respective agents.
        """
        states_a,next_states_a,actions_a,log_probs_a,dones_a,values_a = [],[],[],[],[],[]
        states_b,next_states_b,actions_b,log_probs_b,dones_b,values_b = [],[],[],[],[],[]
        rewards_a = []
        rewards_b = []
        total_rewards = []
        state = env.reset()
        state_a = state[0,:]
        state_b = state[1,:]
        for t in range(200):
            with torch.no_grad():
                action_a,log_prob_a,dist_a,value_a = agent_a.act(state_a)
                action_b,log_prob_b,dist_b,value_b = agent_b.act(state_b)
            action = np.vstack((action_a,action_b))
            next_state,reward,done = env.step(action)
            next_state_a = next_state[0,:]
            next_state_b = next_state[1,:]
            reward_a = reward[0]
            reward_b = reward[1]
            # For ease of multiplication later
            inverse_done = np.logical_not(done).astype(int)
            inverse_dones_a = inverse_done[0]
            inverse_dones_b = inverse_done[1]

            rewards_a.append(reward_a)
            rewards_b.append(reward_b)
            states_a.append(np.expand_dims(state_a,axis=0))
            states_b.append(np.expand_dims(state_b,axis=0))
            next_states_a.append(next_state_a)
            next_states_b.append(next_state_b)
            actions_a.append(action_a)
            actions_b.append(action_b)
            log_probs_a.append(log_prob_a)
            log_probs_b.append(log_prob_b)
            dones_a.append(inverse_dones_a)
            dones_b.append(inverse_dones_b)
            values_a.append(value_a.numpy())
            values_b.append(value_b.numpy())
            state_a = next_state_a
            state_b = next_state_b
            if done.any():
                break
        episode_score = max(np.sum(rewards_a),np.sum(reward_b))
        trajectory_a = (states_a,next_states_a,actions_a,log_probs_a,np.array(dones_a),values_a,np.array(rewards_a))
        trajectory_b = (states_b,next_states_b,actions_b,log_probs_b,np.array(dones_b),values_b,np.array(rewards_b))
        return trajectory_a,trajectory_b,episode_score