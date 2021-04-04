**WARNING: This package is in maintenance mode, please use [Stable-Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) for an up-to-date version. You can find a [migration guide](https://stable-baselines3.readthedocs.io/en/master/guide/migration.html) in SB3 documentation.**

<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

[![CI](https://github.com/hill-a/stable-baselines/workflows/CI/badge.svg)](https://github.com/hill-a/stable-baselines/actions) [![Build Status](https://travis-ci.com/hill-a/stable-baselines.svg?branch=master)](https://travis-ci.com/hill-a/stable-baselines) [![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines.readthedocs.io/en/master/?badge=master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=hill-a/stable-baselines&amp;utm_campaign=Badge_Grade) [![Codacy Badge](https://api.codacy.com/project/badge/Coverage/3bcb4cd6d76a4270acb16b5fe6dd9efa)](https://www.codacy.com/app/baselines_janitors/stable-baselines?utm_source=github.com&utm_medium=referral&utm_content=hill-a/stable-baselines&utm_campaign=Badge_Coverage)

# Stable Baselines

Stable Baselines is a set of improved implementations of reinforcement learning algorithms based on OpenAI [Baselines](https://github.com/openai/baselines/).

You can read a detailed presentation of Stable Baselines in the [Medium article](https://medium.com/@araffin/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82).


These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

**Note: despite its simplicity of use, Stable Baselines (SB) assumes you have some knowledge about Reinforcement Learning (RL).** You should not utilize this library without some practice. To that extent, we provide good resources in the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/rl.html) to get started with RL.

## Main differences with OpenAI Baselines

This toolset is a fork of OpenAI Baselines, with a major structural refactoring, and code cleanups:
- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- More tests & more code coverage
- Additional algorithms: SAC and TD3 (+ HER support for DQN, DDPG, SAC and TD3)


| **Features**                | **Stable-Baselines**              | **OpenAI Baselines**              |
| --------------------------- | --------------------------------- | --------------------------------- |
| State of the art RL methods | :heavy_check_mark: <sup>(1)</sup> | :heavy_check_mark:                |
| Documentation               | :heavy_check_mark:                | :x:                               |
| Custom environments         | :heavy_check_mark:                | :heavy_check_mark:                |
| Custom policies             | :heavy_check_mark:                | :heavy_minus_sign: <sup>(2)</sup> |
| Common interface            | :heavy_check_mark:                | :heavy_minus_sign: <sup>(3)</sup> |
| Tensorboard support         | :heavy_check_mark:                | :heavy_minus_sign: <sup>(4)</sup> |
| Ipython / Notebook friendly | :heavy_check_mark:                | :x:                               |
| PEP8 code style             | :heavy_check_mark:                | :heavy_check_mark: <sup>(5)</sup> |
| Custom callback             | :heavy_check_mark:                | :heavy_minus_sign: <sup>(6)</sup> |

<sup><sup>(1): Forked from previous version of OpenAI baselines, with now SAC and TD3 in addition</sup></sup><br>
<sup><sup>(2): Currently not available for DDPG, and only from the run script. </sup></sup><br>
<sup><sup>(3): Only via the run script.</sup></sup><br>
<sup><sup>(4): Rudimentary logging of training information (no loss nor graph). </sup></sup><br>
<sup><sup>(5): EDIT: you did it OpenAI! :cat:</sup></sup><br>
<sup><sup>(6): Passing a callback function is only available for DQN</sup></sup><br>

## Documentation

Documentation is available online: [https://stable-baselines.readthedocs.io/](https://stable-baselines.readthedocs.io/)

## RL Baselines Zoo: A Collection of 100+ Trained RL Agents

[RL Baselines Zoo](https://github.com/araffin/rl-baselines-zoo). is a collection of pre-trained Reinforcement Learning agents using Stable-Baselines.

It also provides basic scripts for training, evaluating agents, tuning hyperparameters and recording videos.

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

Github repo: https://github.com/araffin/rl-baselines-zoo

Documentation: https://stable-baselines.readthedocs.io/en/master/guide/rl_zoo.html

## Installation

**Note:** Stable-Baselines supports Tensorflow versions from 1.8.0 to 1.14.0. Support for Tensorflow 2 API is planned.

### Prerequisites
Baselines requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows

#### Ubuntu

```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

#### Mac OS X
Installation of system packages on Mac requires [Homebrew](https://brew.sh). With Homebrew installed, run the following:
```bash
brew install cmake openmpi
```

#### Windows 10

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines.readthedocs.io/en/master/guide/install.html#prerequisites).

### Install using pip
Install the Stable Baselines package:
```
pip install stable-baselines[mpi]
```

This includes an optional dependency on MPI, enabling algorithms DDPG, GAIL, PPO1 and TRPO. If you do not need these algorithms, you can install without MPI:
```
pip install stable-baselines
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more details and alternatives (from source, using docker).


## Example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO2 on a cartpole environment:
```python
import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('CartPole-v1')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

env.close()
```

Or just train a model with a one liner if [the environment is registered in Gym](https://github.com/openai/gym/wiki/Environments) and if [the policy is registered](https://stable-baselines.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines import PPO2

model = PPO2('MlpPolicy', 'CartPole-v1').learn(10000)
```

Please read the [documentation](https://stable-baselines.readthedocs.io/) for more examples.


## Try it online with Colab Notebooks !

All the following examples can be executed online using Google colab notebooks:

- [Full Tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [All Notebooks](https://github.com/Stable-Baselines-Team/rl-colab-notebooks)
- [Getting Started](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/stable_baselines_getting_started.ipynb)
- [Training, Saving, Loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/saving_loading_dqn.ipynb)
- [Multiprocessing](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/multiprocessing_rl.ipynb)
- [Monitor Training and Plotting](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/monitor_training.ipynb)
- [Atari Games](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/atari_games.ipynb)
- [RL Baselines Zoo](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/master/rl-baselines-zoo.ipynb)


## Implemented Algorithms

| **Name**            | **Refactored**<sup>(1)</sup> | **Recurrent**      | ```Box```          | ```Discrete```     | ```MultiDiscrete``` | ```MultiBinary```  | **Multi Processing**              |
| ------------------- | ---------------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| ACER                | :heavy_check_mark:           | :heavy_check_mark: | :x: <sup>(5)</sup> | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| ACKTR               | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark:                |
| DDPG                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: <sup>(4)</sup>|
| DQN                 | :heavy_check_mark:           | :x:                | :x:                | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| GAIL <sup>(2)</sup> | :heavy_check_mark:           | :x:                | :heavy_check_mark: |:heavy_check_mark:| :x:                 | :x:                | :heavy_check_mark: <sup>(4)</sup> |
| HER <sup>(3)</sup>  | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :heavy_check_mark:| :x:                               |
| PPO1                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |
| PPO2                | :heavy_check_mark:           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| SAC                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TD3                 | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TRPO                | :heavy_check_mark:           | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: <sup>(4)</sup> |

<sup><sup>(1): Whether or not the algorithm has be refactored to fit the ```BaseRLModel``` class.</sup></sup><br>
<sup><sup>(2): Only implemented for TRPO.</sup></sup><br>
<sup><sup>(3): Re-implemented from scratch, now supports DQN, DDPG, SAC and TD3</sup></sup><br>
<sup><sup>(4): Multi Processing with [MPI](https://mpi4py.readthedocs.io/en/stable/).</sup></sup><br>
<sup><sup>(5): TODO, in project scope.</sup></sup>

NOTE: Soft Actor-Critic (SAC) and Twin Delayed DDPG (TD3) were not part of the original baselines and HER was reimplemented from scratch.

Actions ```gym.spaces```:
 * ```Box```: A N-dimensional box that containes every point in the action space.
 * ```Discrete```: A list of possible actions, where each timestep only one of the actions can be used.
 * ```MultiDiscrete```: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * ```MultiBinary```: A list of possible actions, where each timestep any of the actions can be used in any combination.


## MuJoCo
Some of the baselines examples use [MuJoCo](http://www.mujoco.org) (multi-joint dynamics in contact) physics simulator, which is proprietary and requires binaries and a license (temporary 30-day license can be obtained from [www.mujoco.org](http://www.mujoco.org)). Instructions on setting up MuJoCo can be found [here](https://github.com/openai/mujoco-py)

## Testing the installation
All unit tests in baselines can be run using pytest runner:
```
pip install pytest pytest-cov
make pytest
```

## Projects Using Stable-Baselines

We try to maintain a list of project using stable-baselines in the [documentation](https://stable-baselines.readthedocs.io/en/master/misc/projects.html),
please tell us when if you want your project to appear on this page ;)

## Citing the Project

To cite this repository in publications:

```
@misc{stable-baselines,
  author = {Hill, Ashley and Raffin, Antonin and Ernestus, Maximilian and Gleave, Adam and Kanervisto, Anssi and Traore, Rene and Dhariwal, Prafulla and Hesse, Christopher and Klimov, Oleg and Nichol, Alex and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
  title = {Stable Baselines},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/hill-a/stable-baselines}},
}
```

## Maintainers

Stable-Baselines is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave) and [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli).

**Important Note: We do not do technical support, nor consulting** and don't answer personal questions per email.


## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read **CONTRIBUTING.md** guide first.


## Acknowledgments

Stable Baselines was created in the [robotics lab U2IS](http://u2is.ensta-paristech.fr/index.php?lang=en) ([INRIA Flowers](https://flowers.inria.fr/) team) at [ENSTA ParisTech](http://www.ensta-paristech.fr/en).

Logo credits: [L.M. Tenkes](https://www.instagram.com/lucillehue/)
