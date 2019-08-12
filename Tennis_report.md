# Tennis Report

Initially i tried to solve the environment with a MADDPG agent. 1st version was with an env_wrapper, multiple envs, multiple agents. Each agent fed only its own local observation, each critic fed all actions and observations. I could not get this to work, i thought perhaps it was due to how i was indexing the agents, at the same time i realized the env_wrapper was not needed so i switched to the next version. 2 agents, No env_wrapper. Also unsuccessful. I played around with PPO for a bit, but realized i needed to do something special with the trajectories, since they will typically be of different lengths. So i can't calculate the advantages in one vectorized go. 

At the end i decided to go back to base case - single DDPG agent.
I found some strange behavior, i've always had poor results with DDPG on my local machine, and i've never been clear on what the problem is, despite many hours of digging through debuggers and code. There seemed to be an issue with loading networks directly from the Critic/Actor class, which was bypassed by passing them through? I can't see how this could be a thing, so its probably just some series of weird coincidences. Nonetheless thats what i'm doing for now, because DDPG is so temperamental that it makes me noticably insane!

I really wanted to separate out the actions and observations into separate agents, but since training on the whole batch actually works this is what i have done.

The DDPG model:

1. Seed Replay Buffer with randomly generate trajectories up to the min buffer size
2. Start training for real
3. Each step, calculate the TD error, store the (s,a,r,s') tuple in the Priority Replay Buffer
4. Learn on the samples
   Repeat steps 3-4 until environment is solved.

With a sliding window of 100 the agent solved the env in 1940 episodes.

![Graph](/PPO_performance_mean100.png)

## The Algorithm

**Overview**
Deep Deterministic Policy Gradients is an offline (can learn off-policy, because the gradient step is from the critic), actor+critic method. It uses 4 networks, local actor+critic and target actor+critic. The critics are function aproximations of the Q function. Where they take in an action and state and output a value Q(s,a). In this case, they take in the local states from both agents, and actions from both agents, and output values for both agents given the states/actions. The update step is accomplished by taking the difference between our future target critic projections and current local projections which is the Temporal difference error or TD error. Each step a (s,a,r,s') tuple is stored in the replay buffer. The target networks are updated each step based on the hyperparameter tau, which copies some fraction of the local networks to the target networks. This is to make the target more stationary, as RL algorithms are often unstable in non-stationary environments. Because DDPG is offline, it can learn from any experience in the environment to improve its own policy. 

To augment the ability of a replay buffer to improve performance, i have added the Priority Replay Buffer. This keeps track of the difference between the expected value of the state, and the actual value of the state, and scales the updates from (s,a,r,s') tuples according to how well the agent 'guessed' the reward. The worse the guess, the bigger the update and vice versa.

The actions at each step are chosen deterministically by the policy according to the state. This simplifies the bellman equation significantly, avoiding taking a max over a continuous action space, or descretizing the action space to reduce dimensionality. Because of the deterministic nature, it does not explore by default. Which makes it necessary to add some noise, either to the action itself, or to the parameters generating the action. This also means this is a good strategy if there is always 1 good option, but not if mixed strategies are ideal like in poker for example.

**Details**

1. Adding (s,a,r,s') to the PER

First we must calculate the TD error given by the formula TD_error = reward + gamma*Q'(s,a) - Q(s,a). 
The priority of the tuple is abs(TD_error+epsilon)**alpha. When alpha is 0, all experiences are treated equally. when alpha is 1, all experiences are treated exactly according to the TD_error. (epsilon is added to account for when the TD_error is zero or near zero). We normalize the priorities by the sum of all the priorities. Then we calculate an importance feature to determine the weight of the update.
The importance of a given tuple is the (priority * self.buffer_size)**-self.beta
We then normalize that by the max importance weight of any tuple so that it is always a number between 0 and 1.

2. Acting with noise

**Action noise:**
We forward pass through the network and then add some noise (Gaussian,Orstein-Uhlenbeck)
**Parameter noise:**
We add adaptive noise to each layer of the network. This is tricky because noise will effect each layer differently. This has been demonstrated to be generally superior to action noise. Logically this makes sense as action noise, gives an unexpected outcome to the agent, its primary reason is for exploration so that the agent can see a wide variety of actions and update towards that. Whereas parameter noise means, the agents weights themselves produced the action. So opens up a lot of configurations. It brings to mind if you were trying to play a video game and your actions were noisy it would be super frustrating, and might as well be observing someone else playing the game. But if the controller kept changing, you might still be learning about the connections between each controller configuration and the actions, as well as what the actions do within the environment.

3. The learning step:

_The Critic_
Is updated according to r + Q'(s',a') - Q(s,a). Which corresponds to minimizing the TD error between our current state action value pair, and future state action value pair. Which over time will lead us to update towards the actual return. 

_The Actor_
Is updated according to the the minus mean value the local critic Q of (states,actions). Which means we taking the gradient with respect to maximizing the Q(states,actions) value.

![math](Assets/clip.gif)

Becuase we are training on policy, we must account for the difference between our current network and the previous one which gathered the trajectories, this ratio gives us a way to scale the reward based on the new likelyhood of us choosing the same actions with our current network. So we scale the advantages with the ratio and the clipped ratio, and take the min.

![math](Assets/Lclip.gif)

This is our loss. We then perform gradient ascent on the gradient of the loss to update our actor network.


## Questions

1. My biggest question is also retorical, "WHY do i have such a hard time training DDPG???" I think the answer will be just writing it from scratch again and again, and the added some of the additional updates to the algorithm which will improve performance so i can tell more quickly when it is learning or not.
2. The MADDPG paper mentions training a separate network to model the behavior of other agents. This sounded really interesting, but the Udacity version did not do this as far as i can tell. I guess this is optional?

## Hyperparams:

Hyperparams are all loaded from the config.py file

| Parameter     | Value  | Description                                                |
| ------------- | ------ | ---------------------------------------------------------- |
| SGD_epoch     | 1     | Number of training epoch per iteration                     |
| tmax          | 200    | Number of steps per trajectory                             |
| Gamma         | 0.99   | Discount rate                                              |
| Learning rate | 1e-3   | Critic Learning rate                                       |
| Learning rate | 1e-4   | Actor Learning rate                                       |
| Batch size    | 256 | PER batch size                          |
| Replay size    | 1e6 | PER replay size                          |
| min buffer size   | 1e4 | PER min buffer size                          |
| Beta          | 0.5   | PER hyperparam                                        |
| Beta duration | 1e+5   | PER hyperparam                                        |
| Alpha          | 0.6   | PER priority                                        |

## Future work

One thing i noticed while combing through my program in debug mode, was the the initial state estimates can be quite large (neg or pos). Whereas the rewards are 0.1 or 0. Some possible improvements:

- D4PG
- MADDPG
- normalizing the state input
- learning rate decay

Attributions:
Udacity, @github - Ostamand (hyperparams and sanity checking)
