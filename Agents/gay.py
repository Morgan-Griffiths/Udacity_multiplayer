import numpy as np

def GAY(rewards,values):
    gamma = 0.99
    lam = 0.97
    GAE = np.zeros(len(rewards))
    N = len(rewards)
    for j in range(0,N):
        for i in range(0,N-1):
            sigma = rewards[i] + values[i+1]*gamma - values[i]
            GAE[j] += (gamma*lam)**i * sigma
    return GAE

def reversed_GAE(rewards,values):
    gamma = 0.99
    lam = 0.97
    GAE = np.zeros(len(rewards))
    N = len(rewards)
    l = 0
    for j in range(0,N):
        for i in reversed(range(N-1)):
            sigma = rewards[i] + values[i+1]*gamma - values[i]
            GAE[j] += (gamma*lam)**l * sigma
            l += 1
        l = j
    return GAE


def Vector_GAE(rewards,values):
    N = len(rewards)
    gamma = 0.99
    lam = 0.97

    discounts = gamma**np.arange(N)

