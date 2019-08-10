import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
TODO: Make a continuous space Reinforce policy, where we output a gaussian and calculate the probs based on the mu and sigma.
"""


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def hard_update(source,target):
    for target_param,param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
# PPO
class Policy(nn.Module):
    def __init__(self,device,seed,nS,nA,hidden_dims=(128,128)):
        super(Policy,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_dims = hidden_dims
        self.nA = nA
        self.nS = nS
        self.size = hidden_dims[-1] * hidden_dims[-2]
        self.std = nn.Parameter(torch.zeros(1, nA))
        self.device = device
        
        self.input_layer = nn.Linear(nS,hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(1,len(self.hidden_dims)):
            hidden_layer = nn.Linear(hidden_dims[i-1],hidden_dims[i])
            self.hidden_layers.append(hidden_layer)
        self.actor_output = nn.Linear(hidden_dims[-1],nA)
        self.critic_output = nn.Linear(hidden_dims[-1],1)
        self.reset_parameters()

    def reset_parameters(self):
        self.input_layer.weight.data.uniform_(*hidden_init(self.input_layer))
        for hidden_layer in self.hidden_layers:
            hidden_layer.weight.data.uniform_(*hidden_init(hidden_layer))
        self.critic_output.weight.data.uniform_(-3e-3,3e-3)
            
    def forward(self,state,action=None):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32,device = self.device)
            # x = x.unsqueeze(0)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        # state -> action
        mean = torch.tanh(self.actor_output(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        # Critic state value
        v = self.critic_output(x)
        return action, log_prob, dist.entropy(), v
    
    # Return the action along with the probability of the action. For weighting the reward garnered by the action.
    def act(self,state):
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,dtype=torch.float32) #device = self.device,
            x = x.unsqueeze(0)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        mean = torch.tanh(self.actor_output(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action,log_prob