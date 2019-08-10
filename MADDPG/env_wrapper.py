import numpy as np

class MultiEnv(object):
    def __init__(self,env,nS,K):
        self.base_env = env
        self.K = K
        self.nS = nS
        self.nO = self.K*self.nS # Observation space
        self.envs = [env for _ in range(K)]

    def step(self,actions):
        # Take a step with an action, gather results, sort and concate into obs
        results = [env.step(action) for env,action in zip(self.envs,actions)]
        states = [result[0] for result in results]
        rewards = [result[1] for result in results]
        dones = [result[2] for result in results]
        # Convert into arrays
        # Observations should be 1,K*nS
        obs,rewards,dones = map(lambda x: np.vstack(x), [states,rewards,dones])
        return obs.reshape(1,self.nO),states,rewards,dones

    def reset(self):
        states = [env.reset() for env in self.envs]
        # Stack the states into a single obs
        obs = np.vstack(states).reshape(1,self.nO)
        return obs,states