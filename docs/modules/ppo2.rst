.. _ppo2:

.. automodule:: stable_baselines.ppo2

PPO2
====

The `Proximal Policy Optimization <https://arxiv.org/abs/1707.06347>`_ algorithm combines ideas from A2C (having multiple workers)
and TRPO (it uses a trust region to improve the actor).

The main idea is that after an update, the new policy should be not too far from the old policy.
For that, PPO uses clipping to avoid too large update.

.. note::

  PPO2 is the implementation of OpenAI made for GPU. For multiprocessing, it uses vectorized environments
  compared to PPO1 which uses MPI.

.. note::

  PPO2 contains several modifications from the original algorithm not documented
  by OpenAI: value function is also clipped and advantages are normalized.


Notes
-----

- Original paper: https://arxiv.org/abs/1707.06347
- Clear explanation of PPO on Arxiv Insights channel: https://www.youtube.com/watch?v=5P7I-xPq8u8
- OpenAI blog post: https://blog.openai.com/openai-baselines-ppo/
- ``python -m stable_baselines.ppo2.run_atari`` runs the algorithm for 40M
   frames = 10M timesteps on an Atari game. See help (``-h``) for more
   options.
- ``python -m stable_baselines.ppo2.run_mujoco`` runs the algorithm for 1M
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

Train a PPO agent on `CartPole-v1` using 4 processes.

.. code-block:: python

   import gym

   from stable_baselines.common.policies import MlpPolicy
   from stable_baselines.common import make_vec_env
   from stable_baselines import PPO2

   # multiprocess environment
   env = make_vec_env('CartPole-v1', n_envs=4)

   model = PPO2(MlpPolicy, env, verbose=1)
   model.learn(total_timesteps=25000)
   model.save("ppo2_cartpole")

   del model # remove to demonstrate saving and loading

   model = PPO2.load("ppo2_cartpole")

   # Enjoy trained agent
   obs = env.reset()
   while True:
       action, _states = model.predict(obs)
       obs, rewards, dones, info = env.step(action)
       env.render()

Parameters
----------

.. autoclass:: PPO2
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
    |- cliprange_vf                  |                                                     |
    |- new_tb_log                    |                                                     |
    |- writer                        |                                                     |
    |- t_first_start                 |                                                     |
    |- n_updates                     |                                                     |
    |- mb_obs                        |                                                     |
    |- mb_rewards                    |                                                     |
    |- mb_actions                    |                                                     |
    |- mb_values                     |                                                     |
    |- mb_dones                      |                                                     |
    |- mb_neglogpacs                 |                                                     |
    |- mb_states                     |                                                     |
    |- ep_infos                      |                                                     |
    |- actions                       |                                                     |
    |- values                        |                                                     |
    |- neglogpacs                    |                                                     |
    |- clipped_actions               |                                                     |
    |- rewards                       |                                                     |
    |- infos                         |                                                     |
    +--------------------------------+-----------------------------------------------------+
    |- info                          |From timestep 1                                      |
    |- maybe_ep_info                 |                                                     |
    +--------------------------------+-----------------------------------------------------+

