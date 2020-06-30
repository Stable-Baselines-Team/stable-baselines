.. _a2c:

.. automodule:: stable_baselines.a2c


A2C
====

A synchronous, deterministic variant of `Asynchronous Advantage Actor Critic (A3C) <https://arxiv.org/abs/1602.01783>`_.
It uses multiple workers to avoid the use of a replay buffer.


Notes
-----

-  Original paper:  https://arxiv.org/abs/1602.01783
-  OpenAI blog post: https://openai.com/blog/baselines-acktr-a2c/
-  ``python -m stable_baselines.a2c.run_atari`` runs the algorithm for 40M
   frames = 10M timesteps on an Atari game. See help (``-h``) for more
   options.
-  ``python -m stable_baselines.a2c.run_mujoco`` runs the algorithm for 1M
   frames on a Mujoco environment.

Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️      ✔️
MultiDiscrete ✔️      ✔️
MultiBinary   ✔️      ✔️
============= ====== ===========


Example
-------

Train a A2C agent on `CartPole-v1` using 4 processes.

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy
  from stable_baselines.common import make_vec_env
  from stable_baselines import A2C

  # Parallel environments
  env = make_vec_env('CartPole-v1', n_envs=4)

  model = A2C(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("a2c_cartpole")

  del model # remove to demonstrate saving and loading

  model = A2C.load("a2c_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()

Parameters
----------

.. autoclass:: A2C
  :members:
  :inherited-members:


Callbacks - Accessible Variables
--------------------------------

Depending on initialization parameters and timestep, different variables are accessible.
Variables accessible "From timestep X" are variables that can be accessed when
``self.timestep==X`` in the ``on_step`` function.

      +--------------------------------+-----------------------------------------------------+
      |Variable                        |                                         Availability|
      +================================+=====================================================+
      |- self                          |From timestep 1                                      |
      |- total_timesteps               |                                                     |
      |- callback                      |                                                     |
      |- log_interval                  |                                                     |
      |- tb_log_name                   |                                                     |
      |- reset_num_timesteps           |                                                     |
      |- new_tb_log                    |                                                     |
      |- writer                        |                                                     |
      |- t_start                       |                                                     |
      |- mb_obs                        |                                                     |
      |- mb_rewards                    |                                                     |
      |- mb_actions                    |                                                     |
      |- mb_values                     |                                                     |
      |- mb_dones                      |                                                     |
      |- mb_states                     |                                                     |
      |- ep_infos                      |                                                     |
      |- actions                       |                                                     |
      |- values                        |                                                     |
      |- states                        |                                                     |
      |- clipped_actions               |                                                     |
      |- obs                           |                                                     |
      |- rewards                       |                                                     |
      |- dones                         |                                                     |
      |- infos                         |                                                     |
      +--------------------------------+-----------------------------------------------------+
      |- info                          |From timestep 2                                      |
      |- maybe_ep_info                 |                                                     |
      +--------------------------------+-----------------------------------------------------+
      |- update                        |From timestep ``n_step+1``                           |
      |- rollout                       |                                                     |
      |- masks                         |                                                     |
      |- true_reward                   |                                                     |
      +--------------------------------+-----------------------------------------------------+
      |- value_loss                    |From timestep ``2 * n_step+1``                       |
      |- policy_entropy                |                                                     |
      |- n_seconds                     |                                                     |
      |- fps                           |                                                     |
      +--------------------------------+-----------------------------------------------------+
