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
  import numpy as np

  from stable_baselines.common.vec_env import DummyVecEnv
  from stable_baselines.her.reward_class import ProximalReward
  from stable_baselines.her.utils import stack_obs_goal
  from stable_baselines.her.replay_buffer import FutureHERBuffer
  from stable_baselines.ddpg.noise import NormalActionNoise
  from stable_baselines import DDPG, HER

  env = DummyVecEnv([lambda: gym.make('MountainCarContinuous-v0')])  # The algorithms require a vectorized environment to run

  # the noise objects for DDPG
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))

  # define the reward function, buffer_class and model for HER (+ model parameters)
  model = HER(DDPG, 'MlpPolicy', env, ProximalReward(eps=0.1), buffer_class=FutureHERBuffer, action_noise=action_noise)
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


HER Replay Buffer
-----------------

.. autoclass:: HERBuffer
  :members:


.. autoclass:: EpisodeHERBuffer
  :members:
  :inherited-members:


.. autoclass:: RandomHERBuffer
  :members:
  :inherited-members:


.. autoclass:: FutureHERBuffer
  :members:
  :inherited-members:


Utility functions
-----------------

.. automodule:: stable_baselines.her.utils
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


