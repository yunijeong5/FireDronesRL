"""
Trainer runs an 'environment loop' that runs for exactly one episode 
using the FireDronesEnv class
"""
# import numpy as np
# import pprint

# Import a Trainable (one of RLlib's built-in algorithms):
# We use the PPO algorithm here b/c its very flexible wrt its supported
# action spaces and model types and b/c it learns well almost any problem.
import ray
from ray.rllib.algorithms import ppo
from environment import FireDronesEnv

# Start a new instance of Ray (when running this locally) or
# connect to an already running one (when running this through Anyscale).
ray.init()

# Configure Environment and PPO algorithm
algo = ppo.PPO(
    env=FireDronesEnv,
    config={
        # config to pass to env class
        "env_config": {
            "height": 5,  # grid (forest) size
            "width": 5,
            "prob_tree_plant": 0.5,  # Probability of each cell being a tree
            "num_fires": 2,  # Fire severity: initial number of trees on fire
            "prop_fire_spread": 0.3,  # Probability of fire spreading to a negiboring tree
            "timestep_limit": 100,  # End an episode after this many timesteps
            "num_agents": 3,  # Number of drones
            "agents_vision": 1,  # How far can an agent observe. 1=3x3, 2=5x5, etc.
        }
    },
)
