from gymnasium.spaces import Discrete, MultiDiscrete, Dict
import numpy as np
import random, time
import math
import pygame

from ray.rllib.env.multi_agent_env import MultiAgentEnv

"""
Any environment in RLlib must follow this required class structure:

class YourEnv(SomeEnvClassToInherit):
    def __init__(self, env_config):
        self.action_space = <gymnasium.Space>
        self.observation_space = <gymnasium.Space>
    def reset(self, *, seed=None, options=None):
        return <obs>, <infos>
    def step(self, action):
        return <obs>, <reward: float>, <terminated: bool>, <truncated: bool>, <info: dict>

All other class methods are optional. If YourEnv is a MultiAgentEnv, return values of reset and step should be Dict space.

https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/ 
^ Nice tutorial for creating custom gymnasium environment class
"""


class FireDronesEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

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
        self.fire_coords = set()

        # Probability of fire spreading to a negiboring tree
        self.prob_fire_spread_high = config.get("prob_fire_spread_high", 0.3)
        self.prob_fire_spread_low = config.get("prob_fire_spread_low", 0.05)
        self.prob_fire_spread = np.full(
            (self.height, self.width), self.prob_fire_spread_low
        )
        # Set region with higher fire spread probability (a box at the right side of the grid)
        high_row = self.height / 3
        high_col = math.floor(2 * self.width / 3)
        self.high_spread_area = set()
        for r in range(math.floor(high_row), math.ceil(2 * high_row)):
            for c in range(high_col, self.width):
                self.prob_fire_spread[r, c] = self.prob_fire_spread_high
                self.high_spread_area.add((r, c))

        # End an episode after this many timesteps
        self.timestep_limit = config.get("timestep_limit", 100)

        # Number of drones
        self.num_agents = config.get("num_agents", 5)
        self._agent_ids = set(range(self.num_agents))
        self.agent_pos = {}  # agent positions (row, col)

        # How far each agents can see
        # 1=3x3 square with agent in the middle, 2=5x5 square with agent in the middle
        self.agents_vision = config.get("agents_vision", 1)

        # Action and observation spaces map from agent ids to spacesfor the individual agents.
        ######################################################################################
        # Observation space
        # observation = location (my_row, my_col) + status (of visible cells around me)
        # Ditched "global view", aka all location of fires, bc state space would explode
        # status(neiboring cells): 0=empty, 1=tree, 2=fire, +3 for each drone in the cell (e.g. 6=empty+2 drones; 11: fire+3 drones)
        # Flattened from top-left to bottom-right
        num_visible_cells = (self.agents_vision * 2 + 1) ** 2
        # 2 + 3 * self.num_agents: largest status number (fire + all agents in this cell);
        # 1: for the case where the cell is out of grid (e.g. when drone is at the grid's edges)
        self.CELL_OUTSIDE = (
            2 + 3 * self.num_agents + 1
        )  # code for cells outside grid but in 'visible range'
        state = [self.height, self.width] + (
            [self.CELL_OUTSIDE + 1] * num_visible_cells
        )  # +1 because Discrete's number range is 0...n-1
        self.observation_space = Dict(
            {i: MultiDiscrete(state) for i in range(self.num_agents)}
        )

        # Action space
        # 0=NW, 1=N (up), 2=NE, 3=W (left), 4=spray water (center), 5=E (right), 6=SW, 7=S (down), 8=SE
        self.action_space = Dict({i: Discrete(9) for i in range(self.num_agents)})

        self.pos_update_map = {}  # action number : [row change, col change]
        action_id = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                self.pos_update_map[action_id] = np.array([i, j])
                action_id += 1

        # Reward and penalty constants
        self.TIMESTEP_PENALTY = config.get("time_penalty", -1)
        self.EXTINGUISH_REWARD = config.get("fire_ext_reward", 1)

        # custom performance measure
        self.num_trees = 0  # constant for each episode
        self.num_burnt_trees = 0

        # Rendering variables
        self.do_render = config.get("do_render", False)
        self.window = None
        self.window_size = 512
        self.clock = None
        self.fps = 4

    def reset(self, *, seed=None, options=None):
        """Returns initial observation of next episode."""

        # Reset grid
        self.grid = np.zeros(shape=(self.height, self.width))
        self.num_trees = 0
        self.num_burnt_trees = 0

        # Reset trees: on each cell, trees are planted with `prob_tree_plant`
        for r in range(self.height):
            for c in range(self.width):
                if random.uniform(0, 1) <= self.prob_tree_plant:
                    self.grid[r, c] = 1
                    self.num_trees += 1

        # Reset fires: `num_fires` unique cells are randomly selected and set on fire
        self.fire_coords = set()  # empty set
        fire_count = 0
        while fire_count < self.num_fires:
            r = np.random.choice(range(self.height))  # random num in [0...height-1]
            c = np.random.choice(range(self.width))  # random num in [0...width-1]
            # if a cell's base is tree, set on fire
            if self._is_tree(self.grid[r, c]):
                self.grid[r, c] += 1
                fire_count += 1
                self.fire_coords.add((r, c))
                self.num_burnt_trees += 1

        # Reset positions
        self.agent_pos = {}

        for i in range(self.num_agents):
            # all start from the same cell (e.g. drone storage center)
            self.agent_pos[i] = np.array([0, 0])  # upper left is chosen arbitraryly

        # Update grid status number
        self.grid[0, 0] += 3 * self.num_agents

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        infos = {i: "test in reset" for i in range(self.num_agents)}

        # New rendering env
        if self.do_render:
            self.window = None
            self.clock = None

        return self._get_obs(), infos  # [obs] [infos]

    def step(self, action_dict: dict):
        """
        Returns (next observation, rewards, terminateds, truncateds, infos) after having taken the given actions.

        e.g.
        `action_dict={0: action_for_agent0, 1: action_for_agent1, ...}`
        """
        # increase time step counter by 1
        self.timesteps += 1

        # An episode is done when we reach the time step limit or when there's no more fire
        is_done = self.timesteps >= self.timestep_limit or np.all(self.grid % 3 != 2)

        # Update agent positions based on actions
        rewards = {}
        for agent_id, action in action_dict.items():
            rewards[agent_id] = self._move(agent_id, action)

        # Get observation based on new agent positions
        observations = self._get_obs()

        # Generate a `terminateds` dict (per-agent and total)
        terminateds = dict.fromkeys(range(self.num_agents), is_done)
        terminateds["__all__"] = is_done

        # Fire can spread to its neighboring (8) trees with probability `prop_fire_spread`
        for fr, fc in self.fire_coords.copy():
            neighbor_row = self._get_valid_range(fr, 1, True)
            neighbor_col = self._get_valid_range(fc, 1, False)
            for nr in neighbor_row:
                for nc in neighbor_col:
                    if self._is_tree(self.grid[nr, nc]):
                        if random.uniform(0, 1) <= self.prob_fire_spread[nr, nc]:
                            self.grid[nr, nc] += 1
                            self.fire_coords.add((nr, nc))
                            self.num_burnt_trees += 1

        # Generate a `truncateds` dict (per-agent and total); same as terminated
        truncateds = terminateds.copy()

        # Generate `infos` dict per agent
        infos = (
            {
                i: {"frac_burnt": self.num_burnt_trees / self.num_trees}
                for i in range(self.num_agents)
            }
            if is_done
            else {i: {} for i in range(self.num_agents)}
        )

        # render frame
        if self.do_render:
            self.render()

        if is_done:
            self.close()

        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        )

    def _is_tree(self, cell_state):
        return cell_state % 3 == 1

    def _is_fire(self, cell_state):
        return cell_state % 3 == 2

    def _get_valid_range(self, pos: int, vision: int, is_row: bool):
        if is_row:
            return range(
                max(0, pos - vision),
                min(self.height - 1, pos + vision) + 1,
            )
        else:
            return range(
                max(0, pos - vision),
                min(self.width - 1, pos + vision) + 1,
            )

    def _get_surroundings(self, my_row: int, my_col: int):
        """
        Returns flattened (1D) status codes of surrounding grid cells based on `self.agent_vision`
        """
        agent_obs = [my_row, my_col]
        valid_row = self._get_valid_range(my_row, self.agents_vision, True)
        valid_col = self._get_valid_range(my_col, self.agents_vision, False)

        for r in range(my_row - self.agents_vision, my_row + self.agents_vision + 1):
            for c in range(
                my_col - self.agents_vision, my_col + self.agents_vision + 1
            ):
                if r in valid_row and c in valid_col:
                    agent_obs.append(self.grid[r, c])
                else:
                    agent_obs.append(self.CELL_OUTSIDE)

        return agent_obs

    def _get_obs(self):  # must return dict
        """
        Returns obs dict (agent name to pos+surrounding status) using each
        agent's current x/y-positions (stored in self.agents_pos).
        """
        obs = {}
        for agent_id, (row, col) in self.agent_pos.items():
            obs[agent_id] = np.array(
                self._get_surroundings(row, col), dtype=np.int32
            )  # using int type here is important; one-hot encoding later

        return obs

    def _move(
        self,
        agent_id: int,
        action: int,
    ):
        """
        Moves `agent_id` using given action and update the grid and its reward for the action taken
        Return the agent's reward gained for this action
        """

        # time penalty--longer time to turn off all fire -> lower final reward
        agent_rew = self.TIMESTEP_PENALTY

        # Update fire change
        if action == 4:
            # water on non fire cell -> nothing happens
            if not self._is_fire(
                self.grid[self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]]
            ):
                return agent_rew

            # Fire cell!
            # fire extinguished, no more (burnable) tree in cell
            self.grid[self.agent_pos[agent_id][0], self.agent_pos[agent_id][1]] -= 2
            self.fire_coords.remove(
                (self.agent_pos[agent_id][0], self.agent_pos[agent_id][1])
            )

            # reward for turning off fire
            agent_rew += self.EXTINGUISH_REWARD

            # No need to move for action 8
            return agent_rew

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

        return agent_rew

    def render(self):
        """
        Grid visulaization with Pygame
        """

        # Super simple visualization for quick checks
        # ⬛⬜🟩🟥🌲🔥🤖
        # print(self.grid)
        # for r in range(self.height):
        #     for c in range(self.width):
        #         status = self.grid[r][c]
        #         if status == 0:
        #             print("⬛", end="")
        #         elif status == 1:
        #             print("🟩", end="")
        #         elif status == 2:
        #             print("🟥", end="")
        #         elif status > 2:
        #             print("⬜", end="")
        #     print()
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # size of a single grid cell
        pix_square_width = self.window_size / self.width
        pix_square_height = self.window_size / self.height

        # draw fires (red block)
        for fire in self.fire_coords:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_width * np.array(fire),
                    (pix_square_height, pix_square_width),
                ),
            )

        # draw trees (green block) and mark area with high fire spread prob (yellow border).
        for r in range(self.height):
            for c in range(self.width):
                if self._is_tree(self.grid[r, c]):
                    pygame.draw.rect(
                        canvas,
                        (0, 128, 0),
                        pygame.Rect(
                            pix_square_width * np.array([r, c]),
                            (pix_square_height, pix_square_width),
                        ),
                    )
                if (r, c) in self.high_spread_area:
                    pygame.draw.rect(
                        canvas,
                        (255, 255, 0),
                        pygame.Rect(
                            pix_square_width * np.array([r, c]),
                            (pix_square_height, pix_square_width),
                        ),
                        width=8,
                    )

        # draw agents (black circle)
        for agent in self.agent_pos.values():
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                (agent + 0.5) * pix_square_width,
                pix_square_width / 3,
            )

        # Add gridlines
        for x in range(self.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_width * x),
                (self.window_size, pix_square_width * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_height * x, 0),
                (pix_square_height * x, self.window_size),
                width=3,
            )

        # Copy our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # keep framerate stable
        self.clock.tick(self.fps)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
