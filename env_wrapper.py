import numpy as np

class MultiEnv(object):
    def __init__(self,env,K):
        self.base_env = env
        self.K = K
        self.envs = [env for e in range(K)]

    def step(self,actions):
        # Take a step with an action, gather results, sort and concate into obs
        results = [env.step(action) for env,action in zip(self.envs,actions)]
        states = results[::4]
        rewards = results[1::4]
        dones = results[2::4]
        # Convert into arrays
        obs,rewards,dones = map(lambda x: np.vstack(x), [states,rewards,dones])
        return obs,states,rewards,dones

    def reset(self):
        states = [env.reset() for env in self.envs]
        # Stack the states into a single obs
        obs = np.vstack(states)
        return obs,states