
from unity_env import UnityEnv
import numpy as np
import torch

from ppo import PPO
from config import Config
from train_ppo import train_ppo
from plot import plot

BUFFER_SIZE = 10000
MIN_BUFFER_SIZE = 200
BATCH_SIZE = 25
ALPHA = 0.6 # 0.7 or 0.6
START_BETA = 0.5 # from 0.5-1
END_BETA = 1
QLR = 0.001
ALR = 0.0001
EPSILON = 1
MIN_EPSILON = 0.01
GAMMA = 0.99
TAU = 0.001
L2 = 0.01
N_STEP = 0.95
UPDATE_EVERY = 10
CLIP_NORM = 10

discount_rate = .995
ppo_epsilon = 0.1
ppo_beta = .01
EPISODES = 4000
    
def main(algo):
    seed = 7
    path = 'model_checkpoints/ppo.ckpt'

    # Load the ENV
    # env = UnityEnv(env_file='Environments/Reacher_Linux_one/Reacher.x86_64',no_graphics=True)
    env = UnityEnv(env_file='Environments/Tennis_Linux/Tennis.x86_64',no_graphics=True)

    # number of agents
    num_agents = env.num_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = env.action_size

    # examine the state space 
    state_size = env.state_size
    print('Size of each action: {}, Size of the state space {}'.format(action_size,state_size))
    
    config = Config(algo)
    

    if torch.cuda.is_available():
        try:
            device = torch.device("cuda:0")
            device2 = torch.device("cuda:1")
        except:
            device = torch.device("cuda:0")
    else:
        device = torch.device('cpu')
    try:
        agent_a = PPO(action_size,state_size,seed,device,config)
        agent_b = PPO(action_size,state_size,seed,device2,config)
        print('Double GPU')
    except:
        print('Single GPU')
        agent_a = PPO(action_size,state_size,seed,device,config)
        agent_b = PPO(action_size,state_size,seed,device,config)

    train_ppo(env,[agent_a,agent_b],EPISODES,path)

if __name__ == "__main__":
    algo = 'PPO'
    main(algo)
    