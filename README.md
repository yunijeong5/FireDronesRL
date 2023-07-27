# FireDronesRL

FireDrones is a multi-agent reinforcement learning (MARL) project where multiple agents (drones) are trained to manage wildfire in a 2D grid forest. Drones have local view of grid and collaborate to extinguish the fire quickly.

### Motivation

I hope this project can be used as an example MARL project for beginners in MARL or RLlib. RLlib is undergoing active development, and many existing tutorials online does not accurately reflect the latest version of the library and often includes deprecated code that would cause error. (e.g. Many tutorials don't use the new `ray.rllib.algorithms`)

> **Note**
> This project used Ray 2.5.1 and Gymnasium 0.26.3. Using other versions (especially Ray) may cause error.

_7/25/2023: Project runs on Ray 2.6.1, with more deprecated warnings_

## Problem Description

**Environment Initialization and Updates**: 10 by 10 grid. On each cell, a tree grows with probability X, and fire ignites on K trees. At each time stamp, fire spreads to a neighboring (8 surrounding cells) with probability Y.

^ might have to simplify problem; fixed ignition point maybe

**State**:

**Action**: At time $t$, a drone can move one cell in one of the eight directions (N, NE, E,..., NW), or spray water (=extinguish fire) in its current cell if it's on fire. Drones can enter fire cells and multiple drones may present at the same cell.

**Reward**:
+: extinguished fire
-: time, number of trees on fire
-> might wait to have enough trees to be set on fire. remove +?
-> How to reward individual robot's activity?
->

## Requirements

This project uses the following libraries:

-   [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)
-   [Gymnasium](https://gymnasium.farama.org/)

## Installation

```
conda create -n rllib
conda activate rllib
conda install -c conda-forge "ray-rllib"
```

To see how to install Anaconda, see [here](https://docs.anaconda.com/free/anaconda/install/).

## How to Run

(insert instruction)

**Expected training time**?

### Files

`environment.py`: Defines the wildfire model environment

## Results

Let's set up a separate conda environment to visualizie training process with Tensorboard:

```
conda create -n tb-rllib
conda activate tb-rllib
pip install ray[rllib]
conda install chardet markdown requests tensorboard protobuf=3.20.3
```

A new conda environment is used because libraries needed to run Tensorboard causes conflict in `trainer.py`.

After running `trainer.py`, follow the steps below:

1. Go to ~/ray*results/PPO/PPO_MultiAgentArena*[some key]00000*0[date]*[time]/
2. In that directory, you should see a event.out.... file.
3. Run `tensorboard --logdir .` and head to https://localhost:6006 in your browser.

View detailed results here: /home/yunijeong/ray_results/PPO
To visualize your results with TensorBoard, run: `tensorboard --logdir /home/yunijeong/ray_results/PPO`

## Extending the Project

Here are some additional considerations that can be incoorporated to create more realistic dynamics:

-   Total amount of water each drone can spray is less than or equal to its water capacity
-   Drones must replenish its battery and water
-   Ensure safety distance between fire and drones
-   Add environmental factors (e.g. wind affects fire spread direction)
-   Extendable map with custom grid size
-   etc.

## Useful Tutorials

-   [MultiAgentEnv implementation](https://docs.ray.io/en/latest/_modules/ray/rllib/env/multi_agent_env.html#main-content): The comments are very descriptive and helpful

-   [Example codes found online](./examples/)

-   [Ray Summit 2021 tutorial: a bit outdated but explains MultiAgentEnv well](https://github.com/sven1977/rllib_tutorials/blob/main/ray_summit_2021/tutorial_notebook.ipynb)
