"""
Callbacks can be used for custom metrics and custom postprocessing.
For FireDrones, we want to measure (number of burnt trees / number of trees) at the end of each episode.
The fraction is expected to decrease as the training process progresses, as agents will be able to extinguish fires faster.
"""

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2


class MyCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: EpisodeV2,
        env_index: int,
        **kwargs,
    ):
        episode.custom_metrics["test"] = episode._last_infos[0]["frac_burnt"]
        print(
            f"🔥HELLOOOOO", episode.custom_metrics
        )  # 🔥HELLOOOOO {'test': {'__common__': {}, 0: {0: 'test in step, is_done: True'}, 1: {1: 'test in step, is_done: True'}, 2: {2: 'test in step, is_done: True'}}}
