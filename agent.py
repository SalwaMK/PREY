"""
Q-learning agent module for PREY dungeon crawler.
Implements training, action selection, and model persistence.
"""

import numpy as np
import pickle
import os
from config import ALPHA, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN


class Agent:
    """Q-learning agent for the dungeon crawler environment."""
    def __init__(self):
        self.q_table = {}
        self.epsilon = EPSILON

    def act(self, state):
        """Select an action using epsilon-greedy policy."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.q_table[state])

    def train(self, state, action, reward, next_state):
        """Update Q-table using Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(4)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(4)
        
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q)

    def decay_epsilon(self):
        """Decay epsilon towards minimum value."""
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def save(self, path):
        """Save Q-table to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, path):
        """Load Q-table from pickle file if it exists."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.q_table = pickle.load(f)
