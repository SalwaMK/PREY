"""
Game class for PREY dungeon crawler.
Manages game state and episodes.
"""

from env import Environment
from agent import Agent
from config import *


class Game:
    """Main game class that manages state and episodes."""
    def __init__(self):
        self.env = Environment()
        self.agent = Agent()
        self.agent.load('weights/q_table.pkl')
        self.episode = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False

    def reset(self):
        """Reset the game for a new episode and return initial state."""
        state = self.env.reset()
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        return state

    def step(self, state):
        """Perform one step in the game."""
        action = self.agent.act(state)
        next_state, reward, done = self.env.step(action)
        self.agent.train(state, action, reward, next_state)
        self.step_count += 1
        self.total_reward += reward
        if done or self.step_count >= MAX_STEPS:
            self.done = True
            self.end_episode()
        return next_state, reward, done

    def end_episode(self):
        """Handle end of episode: decay epsilon, save agent, print stats."""
        self.agent.decay_epsilon()
        self.agent.save('weights/q_table.pkl')
        print(f"Episode {self.episode}: Total Reward {self.total_reward:.2f}, Epsilon {self.agent.epsilon:.4f}")
        self.episode += 1
