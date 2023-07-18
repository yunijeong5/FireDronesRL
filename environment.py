import gymnasium
from gymnasium.spaces import Discrete, MultiDiscrete
import numpy as np
import random
import time

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
        self.prob_tree_plant = config.get("prob_tree_plant", 0.5)

        # Fire severity: initial number of trees on fire
        self.num_fires = config.get("num_fires", 5)

        # Probability of fire spreading to a negiboring tree
        self.prob_fire_spread = config.get("prop_fire_spread", 0.3)

        # End an episode after this many timesteps
        self.timestep_limit = config.get("timestep_limit", 100)

        # Number of drones
        self.num_agents = config.get("num_agents", 5)

        # How far each agents can see
        # 1=3x3 square with agent in the middle, 2=5x5 square with agent in the middle
        self.agents_vision = config.get("agents_vision", 1)

        # observation = location(me) + status(visible cells around me) # TODO: sanity check; should the agent know each other's locations
        # Ditched "global view", aka all location of fires, bc state space would explode
        # location(agents): row_pos, col_pos
        # status(neiboring cells): 0=empty, 1=tree, 2=fire, +3 for each drone in the cell (e.g. 6=empty+2 drones; 11: fire+3 drones)
        # Flattened from top-left to bottom-right
        # possible change chain for each init status:
        # 0->[0|3], 1->[1|2|4], 2->[0|2|5], 3->[0|3], 4->[1|4|5], 5->[0|2|3|5]

        num_visible_cells = (self.agents_vision * 2 + 1) ** 2
        # 2 + 3 * self.num_agents: largest status number (fire + all agents in this cell); 1: for the case where the cell is out of grid (NA)
        self.CELL_OUTSIDE = (
            2 + 3 * self.num_agents + 1
        )  # code for cells outside grid but in 'visible range'
        state = [self.height, self.width] + (
            [self.CELL_OUTSIDE + 1] * num_visible_cells
        )  # +1 because Discrete's number range is 0...n-1

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

        # Reward and penalty constants
        self.TIMESTEP_PENALTY = -1
        self.EXTINGUISH_REWARD = 0.1

        # # Reset env
        # self.reset()

    def reset(self):
        """Returns initial observation of next episode."""
        # Reset trees: on each cell, trees are planted with `prob_tree_plant`
        for r in range(self.height):
            for c in range(self.width):
                if random.uniform(0, 1) <= self.prob_tree_plant:
                    self.grid[r, c] = 1

        # Reset fires: `num_fires` unique cells are randomly selected and set on fire
        fire_count = 0
        while fire_count < self.num_fires:
            r = np.random.choice(range(self.height))  # random num in [0...height-1]
            c = np.random.choice(range(self.width))  # random num in [0...width-1]
            # if a cell's base is tree, set on fire
            if self.grid[r, c] % 3 == 1:
                self.grid[r, c] += 1
                fire_count += 1

        # Agent positions
        self.agent_pos = {}

        # Accumulated rewards in this episode
        self.agent_rew = {}

        for i in range(self.num_agents):
            # all start from the same cell (e.g. drone storage center)
            self.agent_pos[i] = [0, 0]  # upper left is chosen arbitraryly

            # reset rewards
            self.agent_rew[i] = 0

        # Update grid status number
        self.grid[0, 0] += 3 * self.num_agents

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
        is_done = self.timesteps >= self.timestep_limit or np.all(self.grid % 3 == 2)

        # Update agent positions based on actions
        for agent_id, action in action_dict.items():
            self._move(agent_id, action)

        # Get observation based on new agent positions
        observations = self._get_obs()

        # Rewards are updated in _move()
        rewards = self.agent_rew.copy()

        # Generate a `done` dict (per-agent and total)
        dones = dict.fromkeys(range(self.num_agents), is_done)
        dones["__all__"] = is_done

        return observations, rewards, dones, {}

    def _get_surroundings(self, my_row: int, my_col: int):
        """
        Returns flattened (1D) status codes of surrounding grid cells based on `self.agent_vision`
        """
        agent_obs = [my_row, my_col]
        valid_row = range(
            max(0, my_row - self.agents_vision),
            min(self.height - 1, my_row + self.agents_vision) + 1,
        )
        valid_col = range(
            max(0, my_col - self.agents_vision),
            min(self.width - 1, my_col + self.agents_vision) + 1,
        )

        for r in range(my_row - self.agents_vision, my_row + self.agents_vision + 1):
            for c in range(
                my_col - self.agents_vision, my_col + self.agents_vision + 1
            ):
                if r in valid_row and c in valid_col:
                    agent_obs.append(self.grid[r, c])
                else:
                    agent_obs.append(self.CELL_OUTSIDE)  # TODO: sanity check

        return agent_obs

    def _get_obs(self):  # must return dict
        """
        Returns obs dict (agent name to pos+surrounding status) using each
        agent's current x/y-positions (stored in self.agents_pos).
        """
        obs = {}
        for agent_id, (row, col) in self.agent_pos.items():
            obs[agent_id] = np.array(self._get_surroundings(row, col))

        return obs

    def _move(
        self,
        agent_id: int,
        action: int,
    ):
        """
        Moves `agent_id` using given action and update the grid and its reward for the action taken
        """

        # time penalty--longer time to turn off all fire -> lower final reward
        self.agent_rew[agent_id] += self.TIMESTEP_PENALTY

        # Update fire change
        if action == 8:  # spray water action
            # fire extinguished, no more (burnable) tree in cell
            self.grid[self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] -= 2

            # reward for turning off fire
            self.agent_rew[agent_id] += self.EXTINGUISH_REWARD

            # No need to move for action 8
            return

        # Update position according to action
        original_r, original_c = self.agent_pos[agent_id][:]
        self.agent_pos[agent_id][0] += self.pos_update_map[action][0]
        self.agent_pos[agent_id][1] += self.pos_update_map[action][1]

        # Collisions: agents CAN be in the same cell. The overlapping cell gets 3 additional "status points" for each drone
        # Hence, underlying condition = grid status % 3

        # Check walls: agents cannot move outside of the grid
        # adjust row
        if self.agent_pos[agent_id][0] < 0:
            self.agent_pos[agent_id][0] = 0
        elif self.agent_pos[agent_id][0] >= self.height:
            self.agent_pos[agent_id][0] = self.height - 1
        # adjust col
        if self.agent_pos[agent_id][1] < 0:
            self.agent_pos[agent_id][1] = 0
        elif self.agent_pos[agent_id][1] >= self.width:
            self.agent_pos[agent_id][1] = self.width - 1

        # Update grid
        self.grid[original_r, original_c] -= 3  # a drone moved out from cell
        self.grid[
            self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]
        ] += 3  # a drome moved into cell

    def render(self):
        """
        Grid visulaization (pygame?) TODO:
        """

        # Super simple implementation for quick checks
        # ⬛⬜🟩🟥🌲🔥🤖
        print(self.grid)
        for r in range(self.height):
            for c in range(self.width):
                status = self.grid[r][c]
                if status == 0:
                    print("⬛", end="")
                elif status == 1:
                    print("🟩", end="")
                elif status == 2:
                    print("🟥", end="")
                elif status > 2:
                    print("⬜", end="")
            print()


env = FireDronesEnv()
print(env.reset())
env.render()
# observations, rewards, dones, infos = env.step(action={...})
