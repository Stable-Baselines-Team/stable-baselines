.. _acer:

.. automodule:: stable_baselines.acer


ACER
====

 `Sample Efficient Actor-Critic with Experience Replay (ACER) <https://arxiv.org/abs/1611.01224>`_ combines
 several ideas of previous algorithms: it uses multiple workers (as A2C), implements a replay buffer (as in DQN),
 uses Retrace for Q-value estimation, importance sampling and a trust region.


Notes
-----

- Original paper: https://arxiv.org/abs/1611.01224
- ``python -m stable_baselines.acer.run_atari`` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (``-h``) for more options.

Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ❌      ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
  from stable_baselines.common import make_vec_env
  from stable_baselines import ACER

  # multiprocess environment
  env = make_vec_env('CartPole-v1', n_envs=4)

  model = ACER(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("acer_cartpole")

  del model # remove to demonstrate saving and loading

  model = ACER.load("acer_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Parameters
----------

.. autoclass:: ACER
  :members:
  :inherited-members:


Callbacks - Accessible Variables 
--------------------------------

Depending on initialization parameters and timestep, different variables are accessible.
Variables accessible from "timestep X" are variables that can be accessed when
``self.timestep==X`` from the ``on_step`` function.

    +--------------------------------+-----------------------------------------------------+
    |Variable                        |                                         Availability|
    +================================+=====================================================+
    |- self                          | From timestep 1                                     |
    |- total_timesteps               |                                                     |
    |- callback                      |                                                     |
    |- log_interval                  |                                                     |
    |- tb_log_name                   |                                                     |
    |- reset_num_timesteps           |                                                     |
    |- new_tb_log                    |                                                     |
    |- writer                        |                                                     |
    |- episode_stats                 |                                                     |
    |- buffer                        |                                                     |
    |- t_start                       |                                                     |
    |- enc_obs                       |                                                     |
    |- mb_obs                        |                                                     |
    |- mb_actions                    |                                                     |
    |- mb_mus                        |                                                     |
    |- mb_dones                      |                                                     |
    |- mb_rewards                    |                                                     |
    |- actions                       |                                                     |
    |- states                        |                                                     |
    |- mus                           |                                                     |
    |- clipped_actions               |                                                     |
    |- obs                           |                                                     |
    |- rewards                       |                                                     |
    |- dones                         |                                                     |
    +--------------------------------+-----------------------------------------------------+
    |- steps                         | From timestep ``n_step+1``                          |
    |- masks                         |                                                     |
    +--------------------------------+-----------------------------------------------------+
    |- names_ops                     | From timestep ``2 * n_step+1``                      |
    |- values_ops                    |                                                     |
    +--------------------------------+-----------------------------------------------------+
    |- samples_number                | After replay_start steps,  when replay_ratio > 0 and|
    |                                | buffer is not None                                  |
    +--------------------------------+-----------------------------------------------------+
