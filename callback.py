"""
Callbacks can be used for custom metrics and custom postprocessing.
For FireDrones, we want to measure (number of burnt trees / number of trees) at the end of each episode.
The fraction is expected to decrease as the training process progresses, as agents will be able to extinguish fires faster.
"""

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode_v2 import EpisodeV2


from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvType,
    ModelGradients,
    ModelWeights,
    MultiAgentPolicyConfigDict,
    PartialAlgorithmConfigDict,
    PolicyID,
    PolicyState,
    SampleBatchType,
    T,
)


class CustomMetricCallback(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        episode: EpisodeV2,
        **kwargs,
    ):
        # fraction of burnt trees at the end of each episode
        episode.custom_metrics["frac_burnt_trees"] = episode._last_infos[0][
            "frac_burnt"
        ]
        # average reward of each agent at then end of an episode
        episode.custom_metrics["mean_agent_reward"] = episode.total_reward / len(
            episode.agent_rewards
        )
