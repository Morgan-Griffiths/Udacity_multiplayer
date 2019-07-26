# Udacity_multiplayer

Submission for completing the Udacity Project

## Implementation

The agent is a combination of 4 add-ons to vanilla DQN.

- Priority Replay
- Double DQN
- Dueling DQN
- Polyak Averaging

Contains the weights of the trained RL bot to solve the problem.
Graphs indicating the progress of the agent and when it solved the problem.

The DQN agent solved the enviroment in 625 steps (Average Reward > 13).

## There are two Environments:

### Tennis

- State space = 24
- Action space = 2 (continuous

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Soccer

Agent Reward Function (dependent):

Striker:
    +1 When ball enters opponent's goal.
    -0.1 When ball enters own team's goal.
    -0.001 Existential penalty.
Goalie:
    -1 When ball enters team's goal.
    +0.1 When ball enters opponents goal.
    +0.001 Existential bonus.

Brains: Two Brain with the following observation/action space:

Vector Observation space: 112 corresponding to local 14 ray casts, each detecting 7 possible object types, along with the object's distance. Perception is in 180 degree view from front of agent.
Vector Action space: (Discrete) One Branch
    Striker: 6 actions corresponding to forward, backward, sideways movement, as well as rotation.
    Goalie: 4 actions corresponding to forward, backward, sideways movement.
Visual Observations: None.

Reset Parameters: None
Benchmark Mean Reward (Striker & Goalie Brain): 0 (the means will be inverse of each other and criss crosses during training)

---

## Project Layout

### Agents

DQN, Priority_DQN

### Buffers

Vanilla ReplayBuffer, PriorityReplayBuffer

### Networks

QNetwork, Dueling_QNetwork

### Main files

train.py
checkpoint.pth

## Installation

Clone the repository.

```
git clone git@github.com:MorGriffiths/Udacity_Navigation.git
cd Udacity_Navigation
```

Create a virtual environment and activate it.

```
python -m venv banana
source banana/bin/activate
```

Install Unity ml-agents.

```
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/.
```

Install the project requirements.

```
pip install -r requirements.txt
```

## Download the Tennis Environment which matches your operating system

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

## Download the Soccer Unity Environment

- [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Place the environment into the Environments folder.
If necessary, inside main.py, change the path to the unity environment appropriately

## Run the project

Each project solution is contained within the notebooks "Navigation.ipynb" and "Navigation_Pixels.ipynb"

Make sure the environment path is correctly set in the desired notebook. Then run the cells as wanted.

## Futher details

The Vector Banana report.md is in the Vector_banana folder. Along with the performance graph and the weights.

Additionally, i tried training visual banana from scratch but likely due to memory constraints it essentially broke in the notebook format. I expect i will be able to train effectively to outside of that. And in addition run some refresh to clear the cache every N epsidoes.

[link](https://medium.com/@C5ipo7i/improving-dqn-cde578df5d73?postPublishedType=initial) A medium article describing the different add-ons i implemented to DQN
