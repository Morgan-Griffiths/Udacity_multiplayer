import sys
import numpy as np
from collections import deque
import time
import torch

from utils.plot import plot

def seed_replay_buffer(env, agent, min_buffer_size):
    obs = env.reset()
    while len(agent.PER) < min_buffer_size:
        # Random actions between 1 and -1
        actions = ((np.random.rand(2,2)*2)-1)
        next_obs,rewards,dones = env.step(actions)
        # reshape
        next_obs = next_obs.reshape(-1)
        obs = obs.reshape(-1)
        actions = torch.from_numpy(actions.reshape(-1)).float()
        # Calculate TD_error
        next_action = agent.actor(next_obs)
        next_value = agent.critic_target(next_obs,next_action)
        target = reward + agent.gamma * next_value * done
        local = agent.critic(obs,action)
        TD_error = target - local
        # Store experience
        agent.PER.add(obs,actions,np.max(rewards),next_obs,np.max(dones),TD_error)
        obs = next_obs
        if dones.any():
            obs = env.reset()
    print('finished replay warm up')

def train_ddpg(env, agent, config):
    episodes,tmax = config.episodes,config.tmax
    tic = time.time()
    means = []
    stds = []
    steps = []
    scores_window = deque(maxlen=100)
    for e in range(1,episodes):
        agent.reset_episode()
        episode_scores = []
        obs = env.reset()
        for t in range(tmax):
            actions = agent.act(obs.reshape(-1))
            next_obs,rewards,dones = env.step(actions.reshape(2,-1))
            # Step agent with reshaped observations
            agent.step(obs.reshape(-1), actions.reshape(-1), np.max(rewards), next_obs.reshape(-1), np.max(dones))
            # Score tracking
            episode_scores.append(np.max(rewards))
            obs = next_obs
            if dones.any():
                steps.append(int(t))
                break
            
        means.append(np.mean(episode_scores))
        stds.append(np.std(episode_scores))
        scores_window.append(np.sum(episode_scores))
        if e % 4 == 0:
            toc = time.time()
            r_mean = np.mean(scores_window)
            r_max = max(scores_window)
            r_min = min(scores_window)
            r_std = np.std(scores_window)
            plot(means,stds,num_agents=2,name=config.name,game='Tennis')
            print("\rEpisode: {} out of {}, Steps {}, Mean steps {:.1f}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,episodes,np.sum(steps),np.mean(steps),r_mean,r_min,r_max,r_std,(toc-tic)/60))
        if np.mean(scores_window) > config.winning_condition:
            print('Env solved!')
            # save scores
            pickle.dump([means,stds], open(str(config.name)+'_scores.p', 'wb'))
            # save policy
            agent.save_weights(config.critic_path,config.actor_path)
            break
    self.env.close()

    

