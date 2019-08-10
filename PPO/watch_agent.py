from unity_env import UnityEnv
import numpy as np

from Agents.ddpg import DDPG
from Agents.ppo import PPO
from Agents.reinforce import REINFORCE
from train_ppo import train
from plot import plot

def main(path='model_checkpoints'):
    seed = 1234
    env = UnityEnv(env_file='Environments/Reacher_Linux_20/Reacher.x86_64',no_graphics=False)

    # number of agents
    num_agents = env.num_agents
    print('Number of agents:', num_agents)

    # size of each action
    action_size = env.action_size

    # examine the state space 
    state_size = env.state_size
    print('Size of each action: {}, Size of the state space {}'.format(action_size,state_size))
    
    path = 'model_checkpoints/ppo.ckpt'
    agent = PPO(env,action_size,state_size,seed)
    agent.load_weights(path)
    rewards = []
    
    state = env.reset()
    for i in range(4000):
        action,_,_,_ = agent.policy(state)
        next_state,reward,done = env.step(action.cpu().numpy())
        # print(next_state,reward,done)
        state = next_state
        rewards.append(np.sum(rewards))
    env.close()
    print("The agent achieved an average score of {:.2f}".format(np.mean(rewards)))

if __name__ == "__main__":
    main()