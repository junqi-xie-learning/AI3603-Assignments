# -*- coding:utf-8 -*-
# Train Dyna-Q in Sokoban environment
import time
import numpy as np
import gym
from agent import DynaQAgent
import gym_sokoban
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

# construct the environment
env = gym.make('Sokoban-hw2-v0')
# get the size of action space
num_actions = env.action_space.n
all_actions = np.arange(num_actions)
# set random seed and make the result reproducible
RANDOM_SEED = 0
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

##### START CODING HERE #####

# construct the intelligent agent.
alpha, gamma, epsilon, planning_time = 1.0, 0.9, 1.0, 10
agent = DynaQAgent(all_actions, alpha, gamma, epsilon)

# start training
for episode in range(100): # training time reduced due to planning stage
    episode_reward = 0
    s = tuple(env.reset())
    # render env. You can comment all render() to turn off the GUI to accelerate training.
    # env.render()
    # agent interacts with the environment
    for iter in range(500):
        a = agent.choose_action(s)
        s_, r, isdone, info = env.step(a)
        s_ = tuple(s_)
        # env.render()
        # update the episode reward
        episode_reward += r
        # agent learns from experience
        agent.learn(s, a, s_, r)
        agent.plan(planning_time)
        s = s_
        if isdone:
            # time.sleep(0.5)
            break
    print('episode:', episode, 'episode_reward:', episode_reward, 'epsilon:', agent.epsilon)
    agent.alpha -= 0.001
    agent.epsilon *= 0.9
print('\ntraining over\n')

# reset for visualization
agent.epsilon = 0
# reset env
s = tuple(env.reset())
# render env
env.render()
while True:
    # choose an action
    a = agent.choose_action(s)
    s_, r, isdone, info = env.step(a)
    s_ = tuple(s_)
    time.sleep(0.5)
    env.render()
    s = s_
    if isdone:
        break

# close the render window after training.
env.close()

##### END CODING HERE #####
