.. _acktr:

.. automodule:: stable_baselines.acktr


ACKTR
=====

`Actor Critic using Kronecker-Factored Trust Region (ACKTR) <https://arxiv.org/abs/1708.05144>`_ uses
Kronecker-factored approximate curvature (K-FAC) for trust region optimization.


Notes
-----

- Original paper: https://arxiv.org/abs/1708.05144
- Baselines blog post: https://blog.openai.com/baselines-acktr-a2c/
- ``python -m stable_baselines.acktr.run_atari`` runs the algorithm for 40M frames = 10M timesteps on an Atari game.
  See help (``-h``) for more options.

Can I use?
----------

-  Recurrent policies: ✔️
-  Multi processing: ✔️
-  Gym spaces:


============= ====== ===========
Space         Action Observation
============= ====== ===========
Discrete      ✔️      ✔️
Box           ✔️       ✔️
MultiDiscrete ❌      ✔️
MultiBinary   ❌      ✔️
============= ====== ===========


Example
-------

.. code-block:: python

  import gym

  from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
  from stable_baselines.common import make_vec_env
  from stable_baselines import ACKTR

  # multiprocess environment
  env = make_vec_env('CartPole-v1', n_envs=4)

  model = ACKTR(MlpPolicy, env, verbose=1)
  model.learn(total_timesteps=25000)
  model.save("acktr_cartpole")

  del model # remove to demonstrate saving and loading

  model = ACKTR.load("acktr_cartpole")

  obs = env.reset()
  while True:
      action, _states = model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      env.render()


Parameters
----------

.. autoclass:: ACKTR
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
    |- self                          |From timestep 1                                      |
    |- total_timesteps               |                                                     |
    |- callback                      |                                                     |
    |- log_interval                  |                                                     |
    |- tb_log_name                   |                                                     |
    |- reset_num_timesteps           |                                                     |
    |- new_tb_log                    |                                                     |
    |- writer                        |                                                     |
    |- tf_vars                       |                                                     |
    |- is_uninitialized              |                                                     |
    |- new_uninitialized_vars        |                                                     |
    |- t_start                       |                                                     |
    |- coord                         |                                                     |
    |- enqueue_threads               |                                                     |
    |- old_uninitialized_vars        |                                                     |
    |- mb_obs                        |                                                     |
    |- mb_rewards                    |                                                     |
    |- mb_actions                    |                                                     |
    |- mb_values                     |                                                     |
    |- mb_dones                      |                                                     |
    |- mb_states                     |                                                     |
    |- ep_infos                      |                                                     |
    |- _                             |                                                     |
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
    |- update                        |From timestep ``n_steps+1``                          |
    |- rollout                       |                                                     |
    |- returns                       |                                                     |
    |- masks                         |                                                     |
    |- true_reward                   |                                                     |
    +--------------------------------+-----------------------------------------------------+
    |- policy_loss                   |From timestep ``2*n_steps+1``                        |
    |- value_loss                    |                                                     |
    |- policy_entropy                |                                                     |
    |- n_seconds                     |                                                     |
    |- fps                           |                                                     |
    +--------------------------------+-----------------------------------------------------+
