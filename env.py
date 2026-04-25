"""
RL environment module for PREY dungeon crawler.
Defines state representation, actions, rewards, and step logic.
"""

import sys
import termios
import tty
import select

from config import GRID_SIZE


class Environment:
    """Reinforcement learning environment for the dungeon crawler."""
    def __init__(self):
        self.hero_x = 0
        self.hero_y = 0
        self.enemy_x = 0
        self.enemy_y = 0

    def reset(self):
        """Reset the environment and return the initial state."""
        self.hero_x = 1
        self.hero_y = 1
        self.enemy_x = 8
        self.enemy_y = 8
        return self._get_state()

    def step(self, action):
        """Execute an enemy action and return (state, reward, done)."""
        self._get_hero_input()

        old_distance = self._manhattan_distance()
        self._move_enemy(action)
        new_distance = self._manhattan_distance()

        done = self._is_caught()
        if done:
            reward = 10.0
        else:
            reward = -0.1
            if new_distance < old_distance:
                reward += 1.0
            elif new_distance > old_distance:
                reward -= 1.0

        return self._get_state(), reward, done

    def _get_hero_input(self):
        """Read keyboard input and update the hero position using WASD."""
        if not sys.stdin.isatty():
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
            else:
                return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if key == "w":
            self.hero_y = max(0, self.hero_y - 1)
        elif key == "s":
            self.hero_y = min(GRID_SIZE - 1, self.hero_y + 1)
        elif key == "a":
            self.hero_x = max(0, self.hero_x - 1)
        elif key == "d":
            self.hero_x = min(GRID_SIZE - 1, self.hero_x + 1)

    def _move_enemy(self, action):
        if action == 0:
            self.enemy_y = max(0, self.enemy_y - 1)
        elif action == 1:
            self.enemy_y = min(GRID_SIZE - 1, self.enemy_y + 1)
        elif action == 2:
            self.enemy_x = max(0, self.enemy_x - 1)
        elif action == 3:
            self.enemy_x = min(GRID_SIZE - 1, self.enemy_x + 1)

    def _is_caught(self):
        """Return True when the enemy catches the hero."""
        return self.hero_x == self.enemy_x and self.hero_y == self.enemy_y

    def _manhattan_distance(self):
        return abs(self.hero_x - self.enemy_x) + abs(self.hero_y - self.enemy_y)

    def _get_state(self):
        return (self.hero_x, self.hero_y, self.enemy_x, self.enemy_y)
