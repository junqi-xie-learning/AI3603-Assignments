# -*- coding:utf-8 -*-
import time
import numpy as np
##### START CODING HERE #####
# This code block is optional. You can import other libraries or define your utility functions if necessary.

##### END CODING HERE #####

"""
Instruction: 
Currently, the following agents are `random` policy.
You need to implement the Q-learning agent, Sarsa agent and Dyna-Q agent in this file.
"""

# ------------------------------------------------------------------------------------------- #

"""TODO: Implement your Sarsa agent here"""
class SarsaAgent(object):
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
    
    def learn(self, state, action, next_state, next_action, reward):
        """learn from experience"""
        next_q = self.get_q(next_state, next_action)
        self.learn_q(state, action, reward + self.gamma * next_q)
    
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

"""TODO: Implement your Q-Learning agent here"""
class QLearningAgent(object):
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
    
    def learn(self, state, action, next_state, reward):
        """learn from experience"""
        max_q = max(self.get_q(next_state, action) for action in self.all_actions)
        self.learn_q(state, action, reward + self.gamma * max_q)
    
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

"""TODO: Implement your Dyna-Q agent here"""
class DynaQAgent(object):
    ##### START CODING HERE #####
    def __init__(self, all_actions):
        """initialize the agent. Maybe more function inputs are needed."""
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with epsilon-greedy algorithm."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None

    ##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #

"""TODO: (optional) Implement RL agent(s) with other exploration methods you have found"""
##### START CODING HERE #####
class RLAgentWithOtherExploration(object):
    """initialize the agent"""
    def __init__(self, all_actions):
        self.all_actions = all_actions
        self.epsilon = 1.0

    def choose_action(self, observation):
        """choose action with other exploration algorithms."""
        action = np.random.choice(self.all_actions)
        return action
    
    def learn(self):
        """learn from experience"""
        time.sleep(0.5)
        print("[INFO] The learning process complete. (ﾉ｀⊿´)ﾉ")
        return True
    
    def your_function(self, params):
        """You can add other functions as you wish."""
        do_something = True
        return None

##### END CODING HERE #####

# ------------------------------------------------------------------------------------------- #