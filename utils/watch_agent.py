import numpy as np
import sys
import torch

sys.path.append('/home/shuza/Code/Udacity_multiplayer')

from config import Config
from DDPG.ddpg_agent import Agent
from unity_env import UnityEnv
from plot import plot
from DDPG.models import Actor,Critic

def main(path='model_checkpoints'):
    seed = 1234
    env = UnityEnv(env_file='../Environments/Tennis_Linux/Tennis.x86_64',no_graphics=False)

    # number of agents
    num_agents = env.num_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = env.action_size

    # examine the state space 
    state_size = env.state_size
    print('Size of each action: {}, Size of the state space {}'.format(action_size,state_size))
    
    config = Config('ddpg')
    path = '/home/shuza/Code/Udacity_multiplayer/DDPG/model_weights/ddpg.ckpt'
    agent = Agent(state_size*2,action_size*2,Actor,Critic,config)
    agent.load_weights(path)
    rewards = []
    
    state = env.reset()
    for i in range(4000):
        action = agent.evaluate(state.reshape(-1))
        next_state,reward,done = env.step(action.reshape(2,-1))
        # print(next_state,reward,done)
        state = next_state
        rewards.append(np.sum(rewards))
        if done.any():
            break
    env.close()
    print("The agent achieved an average score of {:.2f}".format(np.mean(rewards)))

if __name__ == "__main__":
    main()