.. _her:

.. automodule:: stable_baselines.her


HER
====

`Hindsight Experience Replay (HER) <https://arxiv.org/abs/1707.01495>`_


How to use Hindsight Experience Replay
--------------------------------------

HER is a method wrapper that works with Off policy methods (DQN and DDPG for example).
You can still pass the same arguments as with the original methods, the key difference is defining a Reward function (see :ref:`her_rewards`) for the goal exploration.
You will also need to be careful to stack the goal to the observation when using the `predict` function (`stable-baselines.her.utils.stack_obs_goal`).

Example
-------

Train a HER agent on `MountainCarContinuous-v0`.

.. code-block:: python

  import gym

  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.her.reward_class import ProximalReward
  from stable_baselines.her.utils import stack_obs_goal
  from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec
  from stable_baselines import DDPG, HER

  env = DummyVecEnv([lambda: gym.make('MountainCarContinuous-v0')])  # The algorithms require a vectorized environment to run

  param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))

  model = HER(DDPG, 'MlpPolicy', env, ProximalReward(eps=0.1), param_noise=param_noise)  # define the reward function for HER
  model.learn(total_timesteps=25000)
  model.save("her_dqn_mountaincar")

  del model # remove to demonstrate saving and loading

  model = HER.load("her_dqn_mountaincar")

  obs = env.reset()
  while True:
      action, _states = model.predict(stack_obs_goal(obs, [[0.5, 0]]))  # stack the goal to the end of the observation
      obs, rewards, dones, info = env.step(action)
      env.render()



Parameters
----------

.. autoclass:: HER
  :members:
  :inherited-members:

.. _her_rewards:

Reward function
---------------

.. autoclass:: HERRewardFunctions
  :members:


.. autoclass:: ProximalReward
  :members:
  :inherited-members:


Utility functions
-----------------

.. automodule:: utils
  :members:


Custom reward function
----------------------

You can define custom reward function for HER using the `HERRewardFunctions` class:

.. code-block:: python

  import gym

  import numpy as np

  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.her.reward_class import HERRewardFunctions
  from stable_baselines.her.utils import stack_obs_goal
  from stable_baselines import DQN, HER

  class CustomReward(HERRewardFunctions):  # A custom reward function, that will return 1 when the observation has a higher value than the goal
      def get_reward(self, observation, action, goal):
          return 1 if np.any((observation - goal) > 0.1) else -1

  env = DummyVecEnv([lambda: gym.make('MountainCar-v0')])  # The algorithms require a vectorized environment to run

  model = HER(DQN, 'MlpPolicy', env, CustomReward())
  model.learn(total_timesteps=25000)

  obs = env.reset()
  while True:
      action, _states = model.predict(stack_obs_goal(obs, [[0.5, 0]]))  # stack the goal to the end of the observation
      obs, rewards, dones, info = env.step(action)
      env.render()


