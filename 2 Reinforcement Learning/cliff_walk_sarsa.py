# -*- coding:utf-8 -*-
# Train Sarsa in cliff-walking environment
import time
import numpy as np
from gym_gridworld import CliffWalk
from agent import SarsaAgent
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = CliffWalk()
# get the size of action space
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

##### START CODING HERE #####

# construct the intelligent agent.
alpha, gamma, epsilon = 1.0, 0.9, 1.0
agent = SarsaAgent(all_actions, alpha, gamma, epsilon)

# start training
for episode in range(1000):
    # record the reward in an episode
    episode_reward = 0
    # reset env
    s = env.reset()
    a = agent.choose_action(s)
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        # choose an action
        s_, r, isdone, info = env.step(a)
        a_ = agent.choose_action(s_)
        # env.render()
        # update the episode reward
        episode_reward += r
        # agent learns from experience
        agent.learn(s, a, s_, a_, r)
        s, a = s_, a_
        if isdone:
            # time.sleep(0.5)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)
    agent.alpha -= 0.001
    agent.epsilon *= 0.99
print('\ntraining over\n')

# reset for visualization
agent.epsilon = 0
# reset env
s = env.reset()
a = agent.choose_action(s)
# render env
env.render()
while True:
    # choose an action
    s_, r, isdone, info = env.step(a)
    a_ = agent.choose_action(s_)
    time.sleep(0.5)
    env.render()
    s, a = s_, a_
    if isdone:
        break

# close the render window after training.
env.close()

##### END CODING HERE #####
