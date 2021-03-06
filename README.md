# Learning to Navigate with Deep Reinforcement Learning
## Project Details
This project uses deep reinforcement learning to learn to navigate a 3D world, collecting some objects while avoiding others. Specifically, it solves Unity's Bananas environment, where each yellow banana yields a reward of +1 and each blue banana yields a reward of -1. It is the first project of the [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

The state space is a 37-dimensional vector containing the agent's velocity and ray-based projections of nearby objects in its forward direction.

The action space consists of the following four actions:
- Move forward
- Move backward
- Turn left
- Turn right

Episodes begin with the agent in a random location and orientation in a square map, with both types of bananas scattered randomly. As the agent collects bananas, new bananas fall from the sky randomly. Each episode lasts ~300 time steps. The environment is considered solved when an average total score of 13 is obtained over 100 consecutive episodes.

## Getting started
You can set up your environment with the following steps (taken from the Udacity course).

### Installing dependencies
1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__:
    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```
    - __Windows__:
    ```bash
    conda create --name drlnd python=3.6
    activate drlnd
    ```

2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install o
f OpenAI gym.
    - Install the **box2d** environment group by following the instructions [here](https://github.com/openai/
gym#box2d).

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install sever
al dependencies.
```bash
git clone https://github.com/udacity/Value-based-methods.git
cd Value-based-methods/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `d
rlnd` environment.
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-do
wn `Kernel` menu.

### Downloading needed files
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
Then, place the file in the p1_navigation/ folder in the course GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

## Instructions
The jupyter notebook Navigation.ipynb contains all the code needed to train the agent. There are three modes one can run in:
1. Vanilla DQN (`ddqn = False`)
2. Double DQN, selecting actions according to mean(Q1, Q2) (`ddqn = True, ddqn_rand = False`)
3. Double DQN, selecting actions according to random_choice(Q1, Q2) (`ddqn = True, ddqn_rand = True`)

Select a model by specifying the parameters `ddqn` and `ddqn_rand` in the second code block.
