
from unity_env import UnityEnv
import numpy as np

from Agents.ddpg import DDPG
from Agents.ppo import PPO
from Agents.reinforce import REINFORCE
from config import Config
from train_ppo import train_ppo
from train_ddpg import train_ddpg
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
    env = UnityEnv(env_file='Environments/Reacher_Linux_20/Reacher.x86_64',no_graphics=True)

    # number of agents
    num_agents = env.num_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = env.action_size

    # examine the state space 
    state_size = env.state_size
    print('Size of each action: {}, Size of the state space {}'.format(action_size,state_size))
    
    config = Config(algo)
    
    if algo == "DDPG":
        agent = DDPG(seed,action_size,state_size,config)
        train_ddpg(agent,env,UPDATE_EVERY)
    elif algo == 'PPO':
        agent = PPO(env,action_size,state_size,seed,config)
        train_ppo(agent,EPISODES,path)
    else:
        agent = REINFORCE(env,action_size,state_size,seed)
        agent.train()

if __name__ == "__main__":
    algo = 'PPO'
    main(algo)
    