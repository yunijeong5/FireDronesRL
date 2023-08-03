from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from environment import FireDronesEnv

# Get env info
register_env("fire_drones", lambda env_config: FireDronesEnv(env_config))

# Load saved model
trained_fd = Algorithm.from_checkpoint(
    "/home/yunijeong/ray_results/PPO_fire_drones_2023-08-02_17-27-29qgv87dnb_random_policy/checkpoint_000096"
)

# Continue training
# print(trained_fd) # PPO
done = False

while not done:
    result = trained_fd.train()
    print(pretty_print(result))

    if result != None:
        done = True
