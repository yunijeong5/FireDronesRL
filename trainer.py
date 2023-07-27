"""
This is where the training/learning happens.
"""

# Import a Trainable (one of RLlib's built-in algorithms):
# We use the PPO algorithm here b/c its very flexible wrt its supported
# action spaces and model types and b/c it learns well almost any problem.
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from environment import FireDronesEnv
from callback import MyCallback

from examples.multiagent_envs import (
    GuessTheNumberGame,
    MultiAgentCartPole,
    BasicMultiAgent,
)

# Start a new instance of Ray
ray.init()
# register_env("mtcart", lambda env_config: MultiAgentCartPole(env_config))
# tune.run(
#     "PPO",
#     stop={"training_iteration": 5},
#     config={
#         "env": "mtcart",
#         "num_gpus": 0,
#         "num_workers": 1,
#     },
#     checkpoint_at_end=True,
# )
# exit()


# Register custom environment; used in algo config
register_env("fire_drones", lambda env_config: FireDronesEnv(env_config))

# Configure Environment and PPO algorithm
env_config = {
    "height": 5,  # grid (forest) size
    "width": 5,
    "prob_tree_plant": 0.5,  # Probability of each cell being a tree
    "num_fires": 2,  # Fire severity: initial number of trees on fire
    "prob_fire_spread": 0.1,  # Probability of fire spreading to a negiboring tree
    "timestep_limit": 100,  # End an episode after this many timesteps
    "num_agents": 3,  # Number of drones
    "agents_vision": 2,  # How far can an agent observe. 1=3x3, 2=5x5, etc.
    "time_penalty": -1,
    "fire_ext_reward": 1,
}
"""
TODO: increase vision, adjust fire extinguish reward, adjust prob_fire_spread, is_done condition
- vision 2: Didn't do much; slower and increased fluctuation
- more reward for fire (0.1 -> 1): much better rewards and shorter episode lengths! seems more promising than increasing vision
"""

# First create a PPOConfig and add properties to it, like the environment we want to use,
# or the resources we want to leverage for training. After we build the algo from its configuration,
# we can train it for a number of episodes (# of times algo.train() is called) and save the resulting policy periodically (when also.save() is called).


# tune.run(
#     "PPO",
#     stop={"training_iteration": 5},
#     config={
#         "env": "fire_drones",
#         "env_config": {
#             "config": env_config,
#         },
#         "num_gpus": 0,
#         "num_workers": 0,
#         # "multiagent": {
#         #     "policies": policies,
#         #     "policy_mapping_fn": policy_mapping_fn,
#         # },
#         # "framework": "tf",
#     },
#     checkpoint_at_end=True,
# )

# exit()


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=0)
    .resources(num_gpus=0)
    .callbacks(MyCallback)
    .environment(env="fire_drones", env_config=env_config)
    .build()
)

for i in range(20):
    result = algo.train()
    print(pretty_print(result))
    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"checkpoint saved in directory {checkpoint_dir}")
