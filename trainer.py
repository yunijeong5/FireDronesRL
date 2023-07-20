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
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

# from ray import air, tune
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent

from fd_environment import FireDronesEnv

# TODO: test with official example code
from examples.multiagent_envs import (
    GuessTheNumberGame,
    MultiAgentCartPole,
    BasicMultiAgent,
)
from examples.singleagent_env import GridWorldEnv


# register_env("test", lambda x: GridWorldEnv())
# register_env("basic", lambda x: BasicMultiAgent())


# algo = PPOConfig().environment(env="basic").build()
# result = algo.train()
# print(pretty_print(result))

# exit()

# for i in range(10):
#     result = algo.train()
#     print(pretty_print(result))
#     if i % 5 == 0:
#         checkpoint_dir = algo.save()
#         print(f"checkpoint saved in directory {checkpoint_dir}")

# exit()


# Start a new instance of Ray (when running this locally) or
# connect to an already running one (when running this through Anyscale).
ray.init()

# # Register custom environment
register_env("fire_drones", lambda env_config: FireDronesEnv(env_config))
# forced_multi = make_multi_agent("fire_drones")

# Configure Environment and PPO algorithm
# config to pass to env class
env_config = {
    "height": 5,  # grid (forest) size
    "width": 5,
    "prob_tree_plant": 0.5,  # Probability of each cell being a tree
    "num_fires": 2,  # Fire severity: initial number of trees on fire
    "prob_fire_spread": 0.3,  # Probability of fire spreading to a negiboring tree
    "timestep_limit": 100,  # End an episode after this many timesteps
    "num_agents": 3,  # Number of drones
    "agents_vision": 1,  # How far can an agent observe. 1=3x3, 2=5x5, etc.
}

# First create a PPOConfig and add properties to it, like the environment we want to use,
# or the resources we want to leverage for training. After we build the algo from its configuration,
# we can train it for a number of episodes (here 10) and save the resulting policy periodically (here every 5 episodes).
algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=0)
    # .resources(num_gpus=0)
    .environment(env="fire_drones", env_config=env_config)
    .build()
)
result = algo.train()

exit()

for i in range(10):
    result = algo.train()
    print(pretty_print(result))
    if i % 5 == 0:
        checkpoint_dir = algo.save()
        print(f"checkpoint saved in directory {checkpoint_dir}")


# algo = ppo.PPO(
#     env=FireDronesEnv,
# config={
#     # config to pass to env class
#     "env_config": {
#         "height": 5,  # grid (forest) size
#         "width": 5,
#         "prob_tree_plant": 0.5,  # Probability of each cell being a tree
#         "num_fires": 2,  # Fire severity: initial number of trees on fire
#         "prop_fire_spread": 0.3,  # Probability of fire spreading to a negiboring tree
#         "timestep_limit": 100,  # End an episode after this many timesteps
#         "num_agents": 3,  # Number of drones
#         "agents_vision": 1,  # How far can an agent observe. 1=3x3, 2=5x5, etc.
#     }
#     },
# )


# num_rollout_workers: int | None = NotProvided,
#     num_envs_per_worker: int | None = NotProvided,
#     create_env_on_local_worker: bool | None = NotProvided,
#     sample_collector: type[SampleCollector] | None = NotProvided,
#     sample_async: bool | None = NotProvided,
#     enable_connectors: bool | None = NotProvided,
#     rollout_fragment_length: int | str | None = NotProvided,
#     batch_mode: str | None = NotProvided,
#     remote_worker_envs: bool | None = NotProvided,
#     remote_env_batch_wait_ms: float | None = NotProvided,
#     validate_workers_after_construction: bool | None = NotProvided,
#     preprocessor_pref: str | None = NotProvided,
#     observation_filter: str | None = NotProvided,
#     synchronize_filter: bool | None = NotProvided,
#     compress_observations: bool | None = NotProvided,
#     enable_tf1_exec_eagerly: bool | None = NotProvided,
#     sampler_perf_stats_ema_coef: float | None = NotProvided,
#     horizon: int = DEPRECATED_VALUE,
#     soft_horizon: int = DEPRECATED_VALUE,
#     no_done_at_end: int = DEPRECATED_VALUE,
#     ignore_worker_failures: int = DEPRECATED_VALUE,
#     recreate_failed_workers: int = DEPRECATED_VALUE,
#     restart_failed_sub_environments: int = DEPRECATED_VALUE,
#     num_consecutive_worker_failures_tolerance: int = DEPRECATED_VALUE,
#     worker_health_probe_timeout_s: int = DEPRECATED_VALUE,
#     worker_restore_timeout_s: int = DEPRECATED_VALUE
# ) -> AlgorithmConfig
