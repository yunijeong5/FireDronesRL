"""
This is where the training/learning happens.
"""

# Import a Trainable (one of RLlib's built-in algorithms):
# We use the PPO algorithm here b/c its very flexible wrt its supported
# action spaces and model types and b/c it learns well almost any problem.
import ray
from ray.rllib.algorithms.ppo import (
    PPOConfig,
    PPOTorchPolicy,
)
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.examples.policy.random_policy import RandomPolicy

from environment import FireDronesEnv
from callback import CustomMetricCallback


# Start a new instance of Ray
ray.init()

# Register custom environment; used in algo config
register_env("fire_drones", lambda env_config: FireDronesEnv(env_config))

# Configure Environment and PPO algorithm
env_config = {
    "height": 10,  # grid (forest) size (>= 4)
    "width": 10,
    "prob_tree_plant": 0.5,  # Probability of each cell being a tree
    "num_fires": 2,  # Fire severity: initial number of trees on fire
    "prob_fire_spread_high": 0.2,  # Probability of fire spreading to a negiboring tree in high-spread region
    "prob_fire_spread_low": 0.05,  # Probability of fire spreading to a negiboring tree in low-spread region
    "timestep_limit": 100,  # End an episode after this many timesteps
    "num_agents": 10,  # Number of drones
    "agents_vision": 1,  # How far can an agent observe. 1=3x3, 2=5x5, etc.
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
env = FireDronesEnv()

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=0)
    .resources(num_gpus=0)
    .callbacks(CustomMetricCallback)
    .environment(env="fire_drones", env_config=env_config)
    .multi_agent(
        policies={
            "ppo": PolicySpec(policy_class=PPOTorchPolicy),
            "random": PolicySpec(
                policy_class=RandomPolicy
            ),  # Simple baseline test: PPO vs Random
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: [
            "ppo",
            "random",
        ][agent_id % 2],
        policies_to_train=["ppo"],
    )
    .rl_module(
        rl_module_spec=MultiAgentRLModuleSpec(
            module_specs={
                "learnable_policy": SingleAgentRLModuleSpec(),
                "random": SingleAgentRLModuleSpec(module_class=RandomRLModule),
            }
        ),
    )
    .build()
)

# algo = (
#     AlgorithmConfig(algo_class="PPO")
#     .rollouts(num_rollout_workers=0)
#     .resources(num_gpus=0)
#     .callbacks(CustomMetricCallback)
#     .environment(env="fire_drones", env_config=env_config)
#     .multi_agent(
#         policies={
#             "ppo_policy": PolicySpec(policy_class=PPOTF1Policy),
#             "random": PolicySpec(policy_class=RandomPolicy),
#         },
#         policy_mapping_fn=lambda agent_id, *args, **kwargs: [
#             "ppo_policy",
#             "random",
#         ][agent_id % 2],
#         policies_to_train=["ppo_policy"],
#     )
#     .build()
# )


for i in range(6):
    result = algo.train()
    print(pretty_print(result))
    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"checkpoint saved in directory {checkpoint_dir}")


ray.shutdown()
