from unity_env import UnityEnv
import numpy as np
import gym

from config import Config
from train import train
from unity_env import UnityEnv
from env_wrapper import MultiEnv
from maddpg import MultiAgent
from plot import plot

def main(path='model_checkpoints'):
    # seed = 1234
    ### For unity ###
    env = UnityEnv(env_file='Environments/Reacher_Linux_20/Reacher.x86_64',no_graphics=False)

    # number of agents
    num_agents = env.num_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = env.action_size

    # examine the state space 
    state_size = env.state_size
    print('Size of each action: {}, Size of the state space {}'.format(action_size,state_size))

    ### For gym ###
    # K = 2
    # env = gym.make('MountainCarContinuous-v0')
    # nS = env.observation_space.shape[0]
    # nA = env.action_space.shape[0]
    # K_envs = MultiEnv(env,nS,K)
    ddpg_config = Config('maddpg')
    maddpg = MultiAgent(env,state_size,action_size,ddpg_config)
    maddpg.load_weights(ddpg_config.critic_path,ddpg_config.actor_path)
    maddpg.evaluate()

if __name__ == "__main__":
    main()