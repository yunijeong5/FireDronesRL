import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import random

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class FireDronesEnv(MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        # Dimentions of the grid
        self.height = config.get("height", 10)  # number of rows
        self.width = config.get("width", 10)  # number of columns

        # Make grid
        self.grid = np.zeros(shape=(self.height, self.width))

        # Probability of each cell being a tree
        self.prob_tree_plant = config.get(
            "prob_tree_plant", 0.5
        )  # TODO: how much randomness?

        # Fire severity: initial number of trees on fire
        self.num_fires = config.get("num_fires", 5)

        # Probability of fire spreading to a negiboring tree
        self.prob_fire_spread = config.get("prop_fire_spread", 0.3)

        # End an episode after this many timesteps
        self.timestep_limit = config.get("timestep_limit", 100)  # TODO: time limit

        # Number of drones
        self.num_agents = config.get("num_agents", 5)

        # observation = location(agents) + status(grid) OR location(agents) + location(fires) # TODO: sanity check
        # location(agents): each number is discredt representation of a cell coordinate (0~width*height-1)
        # status(grid): 0=no tree, 1=tree(not on fire), 2=tree(on fire)
        # possible change chain for each init status: 0->0, 1->2, 1->2->0, 2->0

        # TODO: how to express location of fire? width*height number of fires in total. ^kinda like boolean flag....
        # Challenge: number of fires is unknown at the beginning
        num_cells = self.width * self.height
        state = [num_cells] * self.num_agents + [3] * num_cells

        self.observation_space = MultiDiscrete(state)

        # Agent actions
        # 0=N (up), 1=NE, 2=E (right), 3=SE, 4=S (down), 5=SW, 6=W (left), 7=NW, 8=spray water
        self.action_space = Discrete(9)
        self.pos_update_map = {  # action number : [row change, col change]
            0: [-1, 0],
            1: [-1, 1],
            2: [0, 1],
            3: [1, 1],
            4: [1, 0],
            5: [1, -1],
            6: [0, -1],
            7: [-1, -1],
            8: [0, 0],
        }

        # Reset env
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Agent positions
        self.agent_pos = {}

        # Accumulated rewards in this episode
        self.agent_rew = {}

        for i in range(self.num_agents):
            # all start from upper left corner (drone storage center)
            self.agent_pos[i] = [0, 0]

            # reset rewards
            self.agent_rew[i] = 0.0

        # Reset trees: on each cell, trees are planted with `prob_tree_plant`
        for r in range(self.height):
            for c in range(self.width):
                if random.uniform(0, 1) <= self.prob_tree_plant:
                    self.grid[r][c] = 1

        # Reset fires: `num_fires` unique cells are randomly selected and set on fire
        fire_rows = np.random.choice(
            range(self.height), size=self.num_fires, replace=False
        )
        fire_cols = np.random.choice(
            range(self.width), size=self.num_fires, replace=False
        )
        for r, c in zip(fire_rows, fire_cols):
            self.grid[r][c] = 2

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        return self._get_obs()

    def step(self, action_dict: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """

        # increase time step counter by 1
        self.timesteps += 1

        # An episode is done when we reach the time step limit or when there's no more fire
        prev_state = self._get_obs()
        is_done = (
            self.timesteps >= self.timestep_limit
            or sum(  # TODO: use np.sum if prev_state is numpy array
                prev_state[self.num_agents :]
            )
            == 0
        )

        for agent_id, action in action_dict:
            self._move(agent_id, action)

        # TODO: continue implementation; not done yet!
        # Get observation based on new agent positions

    def _get_obs(self):  # must return dict
        """
        Returns obs dict (agent name to discrete-pos tuple) using each
        agent's current x/y-positions.
        """
        obs = {}
        agent_pos_discrete = {}
        for agent_id in range(self.num_agents):
            # each value: MultiDiscrete(...) as defined in __init__
            agent_pos_discrete[agent_id] = (
                self.agent_pos[agent_id][0] * self.width
                + self.agent_pos[agent_id][1] % self.width
            )

        # [locations..., fires....]

        # TODO: Trees?? Fires??

        return obs

    def _move(
        self,
        agent_id: int,
        action: int,
    ):
        """
        Moves `agent_id` from `coords` using given action and returns a resulting events dict:
        """

        # Update position according to action
        self.agent_pos[agent_id][0] += self.pos_update_map[action][0]
        self.agent_pos[agent_id][1] += self.pos_update_map[action][1]

        # Update fire change
        if action == 8:  # spray water action
            # fire extinguished, no more (burnable) tree in cell
            self.grid[self.agent_pos[agent_id][0]][self.agent_pos[agent_id][1]] = 0

    def print_grid(self):
        print(self.grid)


# env = FireDronesEnv()
# env.print_grid()
