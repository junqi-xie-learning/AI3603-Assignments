# -*- coding:utf-8 -*-
import numpy as np

"""
Instruction: 
You need to implement the Q-learning agent, Sarsa agent and Dyna-Q agent in this file.
"""

##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.
class Agent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions, alpha, gamma, epsilon):
        """initialize the agent"""
        self.all_actions = all_actions
        self.q_values = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm"""
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.all_actions)
        else:
            q_values = [self.get_q(observation, action) for action in self.all_actions]
            action = self.all_actions[q_values.index(max(q_values))]
        return action

    def get_q(self, state, action):
        """get Q-value from self.q_values"""
        self.q_values.setdefault((state, action), 0)
        return self.q_values[(state, action)]
    
    def learn_q(self, state, action, reward):
        """update Q-value according to reward"""
        self.q_values[(state, action)] = (1 - self.alpha) * self.get_q(state, action) + \
            self.alpha * reward
##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Sarsa agent here"""
class SarsaAgent(Agent):
    ##### START CODING HERE #####
    def learn(self, state, action, next_state, next_action, reward):
        """learn from experience"""
        next_q = self.get_q(next_state, next_action)
        self.learn_q(state, action, reward + self.gamma * next_q)
    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Q-Learning agent here"""
class QLearningAgent(Agent):
    ##### START CODING HERE #####   
    def learn(self, state, action, next_state, reward):
        """learn from experience"""
        max_q = max(self.get_q(next_state, action) for action in self.all_actions)
        self.learn_q(state, action, reward + self.gamma * max_q)
    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Dyna-Q agent here"""
class DynaQAgent(QLearningAgent):
    ##### START CODING HERE #####
    def __init__(self, all_actions, alpha, gamma, epsilon):
        """initialize the agent"""
        super().__init__(all_actions, alpha, gamma, epsilon)
        self.model = {}

    def learn(self, state, action, next_state, reward):
        """learn from experience"""
        super().learn(state, action, next_state, reward)
        self.model[(state, action)] = (next_state, reward)

    def plan(self, time):
        """interact with the model"""
        for _ in range(time):
            keys = list(self.model.keys())
            index = np.random.choice(range(len(keys)))
            state, action = keys[index]
            next_state, reward = self.model[(state, action)]
            self.learn(state, action, next_state, reward)
    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #