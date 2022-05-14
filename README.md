# Learning to Navigate with Deep Reinforcement Learning
## Project Details
This project uses deep reinforcement learning to learn to navigate a 3D world, collecting some objects while avoiding others. Specifically, it solves Unity's Bananas environment, where each yellow banana yields a reward of +1 and each blue banana yields a reward of -1.

The state space is a 37-dimensional vector containing the agent's velocity and ray-based projections of nearby objects in its forward direction.

The action space consists of the following four actions:
- Move forward
- Move backward
- Turn left
- Turn right

Episodes begin with the agent in a random location and orientation in a square map, with both types of bananas scattered randomly. As the agent collects bananas, new bananas fall from the sky randomly. Each episode lasts ~300 time steps. The environment is considered solved when an average total score of 13 is obtained over 100 consecutive episodes.

## Getting started
### Installing dependencies
### Downloading needed files

## Instructions
The jupyter notebook Collect_Bananas.ipynb contains all the code needed to train the agent. There are three modes one can run in:
