import numpy as np
from collections import deque
import time
from plot import plot

from noise import initialize_N, OUnoise

def reshape_trajectory(state,action,reward,next_state,done):
    return (state.reshape(1,20,33),action.reshape(1,20,4),reward.reshape(1,20),next_state.reshape(1,20,33),done.reshape(1,20))


def train_ddpg(agent,env,SGD_update,epsilon=1,noise_decay=50,n_episodes=100, tmax=500):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
        tmax (int): maximum number of timesteps per episode
        SGD_update: how many updates per trajectory
        epsilon: discounts the noise over time so that the noise trends to 0
        Instead of updating target every (int) steps, using 'soft' updating of .1 to gradually merge the networks
    """
    tic = time.time()
    name = "DDPG"
    means = []
    stds = []
    scores_window = deque(maxlen=100)
    index = 0
    
    # N = initialize_N(n_episodes)
    N = OUnoise(agent.nA,7)
    # Solve env
    for e in range(1,n_episodes):
        # reset the environment
        state = env.reset()
        # epsilon -= epsilon/noise_decay
        episode_score = [] 
        for t in range(1,tmax):
            # noise = max(N[e] * epsilon,0)
            action = agent.act(state,N.sample())
            # print('action',action,action.shape)
            next_state,reward,done = env.step(action)
            episode_score.append(reward)
            trajectory = reshape_trajectory(state,action,reward,next_state,done)
            agent.add(trajectory)
            if len(agent.R) > agent.min_buffer_size and t % 20 == 0:
                for _ in range(SGD_update):
                    agent.step()
            state = next_state
            if done.any():
                break
                
        score = np.sum(episode_score)
        scores_window.append(score)
        means.append(np.mean(scores_window))
        stds.append(np.std(scores_window))

        if e % 5 == 0:
            toc = time.time()
            r_mean = np.mean(scores_window)
            r_max = max(scores_window)
            r_min = min(scores_window)
            r_std = np.std(scores_window)
            plot(name,means,stds)
            print("\rEpisode: {} out of {}, Steps {}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,n_episodes,int(e*tmax*20),r_mean,r_min,r_max,r_std,(toc-tic)/60))
            if r_mean > 30:
                print('Env solved!')
                # save scores
                pickle.dump([means,stds], open('ddpg_scores.p', 'wb'))
                # save policy
                agent.save_weights(path)
                break