"""
Configuration module for PREY dungeon crawler RL game.
Contains all constants: grid parameters, colors, FPS, and Q-learning hyperparameters.
"""

# Grid parameters
GRID_SIZE = 10
CELL_SIZE = 60

# Rendering
FPS = 10

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)  # Hero
RED = (255, 0, 0)  # Enemy
GRAY = (128, 128, 128)  # Wall

# Q-learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.95  # Discount factor
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995  # Decay rate per episode
EPSILON_MIN = 0.05  # Minimum exploration rate

# Training parameters
EPISODES = 5000
MAX_STEPS = 200
