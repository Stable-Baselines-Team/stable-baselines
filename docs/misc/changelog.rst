.. _changelog:

Changelog
==========

For download links, please look at `Github release page <https://github.com/hill-a/stable-baselines/releases>`_.


Release 2.10.2 (2021-04-05)
---------------------------

.. warning::

    This package is in maintenance mode, please use `Stable-Baselines3
    (SB3)`_ for an up-to-date version. You can find a `migration guide`_ in
    SB3 documentation.


.. _Stable-Baselines3 (SB3): https://github.com/DLR-RM/stable-baselines3
.. _migration guide: https://stable-baselines3.readthedocs.io/en/master/guide/migration.html


Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- EvalCallback now works also for recurrent policies (@mily20001)

Bug Fixes:
^^^^^^^^^^
- Fixed calculation of the log probability of Diagonal Gaussian distribution
  when using ``action_probability()`` method (@SVJayanthi, @sunshineclt)
- Fixed docker image build (@anj1)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Faster tests, switched to GitHub CI

Documentation:
^^^^^^^^^^^^^^
- Added stable-baselines-tf2 link on Projects page. (@sophiagu)
- Fixed a typo in ``stable_baselines.common.env_checker.check_env`` (@OGordon100)

Release 2.10.1 (2020-08-05)
---------------------------

**Bug fixes release**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``render()`` method of ``VecEnvs`` now only accept one argument: ``mode``

New Features:
^^^^^^^^^^^^^
- Added momentum parameter to A2C for the embedded RMSPropOptimizer (@kantneel)
- ActionNoise is now an abstract base class and implements ``__call__``, ``NormalActionNoise`` and ``OrnsteinUhlenbeckActionNoise`` have return types (@PartiallyTyped)
- HER now passes info dictionary to compute_reward, allowing for the computation of rewards that are independent of the goal (@tirafesi)

Bug Fixes:
^^^^^^^^^^
- Fixed DDPG sampling empty replay buffer when combined with HER  (@tirafesi)
- Fixed a bug in ``HindsightExperienceReplayWrapper``, where the openai-gym signature for ``compute_reward`` was not matched correctly (@johannes-dornheim)
- Fixed SAC/TD3 checking time to update on learn steps instead of total steps (@PartiallyTyped)
- Added ``**kwarg`` pass through for ``reset`` method in ``atari_wrappers.FrameStack`` (@PartiallyTyped)
- Fix consistency in ``setup_model()`` for SAC, ``target_entropy`` now uses ``self.action_space`` instead of ``self.env.action_space`` (@PartiallyTyped)
- Fix reward threshold in ``test_identity.py``
- Partially fix tensorboard indexing for PPO2 (@enderdead)
- Fixed potential bug in ``DummyVecEnv`` where ``copy()`` was used instead of ``deepcopy()``
- Fixed a bug in ``GAIL`` where the dataloader was not available after saving, causing an error when using ``CheckpointCallback``
- Fixed a bug in ``SAC`` where any convolutional layers were not included in the target network parameters.
- Fixed ``render()`` method for ``VecEnvs``
- Fixed ``seed()``` method for ``SubprocVecEnv``
- Fixed a bug ``callback.locals`` did not have the correct values (@PartiallyTyped)
- Fixed a bug in the ``close()`` method of ``SubprocVecEnv``, causing wrappers further down in the wrapper stack to not be closed. (@NeoExtended)
- Fixed a bug in the ``generate_expert_traj()`` method in ``record_expert.py`` when using a non-image vectorized environment (@jbarsce)
- Fixed a bug in CloudPickleWrapper's (used by VecEnvs) ``__setstate___`` where loading was incorrectly using ``pickle.loads`` (@shwang).
- Fixed a bug in ``SAC`` and ``TD3`` where the log timesteps was not correct(@YangRui2015)
- Fixed a bug where the environment was reset twice when using ``evaluate_policy``


Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Added ``version.txt`` to manage version number in an easier way
- Added ``.readthedocs.yml`` to install requirements with read the docs
- Added a test for seeding ``SubprocVecEnv``` and rendering

Documentation:
^^^^^^^^^^^^^^
- Fix typos (@caburu)
- Fix typos in PPO2 (@kvenkman)
- Removed ``stable_baselines\deepq\experiments\custom_cartpole.py`` (@aakash94)
- Added Google's motion imitation project
- Added documentation page for monitor
- Fixed typos and update ``VecNormalize`` example to show normalization at test-time
- Fixed ``train_mountaincar`` description
- Added imitation baselines project
- Updated install instructions
- Added Slime Volleyball project (@hardmaru)
- Added a table of the variables accessible from the ``on_step`` function of the callbacks for each algorithm (@PartiallyTyped)
- Fix typo in README.md (@ColinLeongUDRI)
- Fix typo in gail.rst (@roccivic)

Release 2.10.0 (2020-03-11)
---------------------------

**Callback collection, cleanup and bug fixes**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``evaluate_policy`` now returns the standard deviation of the reward per episode
  as second return value (instead of ``n_steps``)
- ``evaluate_policy`` now returns as second return value a list of the episode lengths
  when ``return_episode_rewards`` is set to ``True`` (instead of ``n_steps``)
- Callback are now called after each ``env.step()`` for consistency (it was called every ``n_steps`` before
  in algorithm like ``A2C`` or ``PPO2``)
- Removed unused code in ``common/a2c/utils.py`` (``calc_entropy_softmax``, ``make_path``)
- **Refactoring, including removed files and moving functions.**

   - Algorithms no longer import from each other, and ``common`` does not import from algorithms.
   - ``a2c/utils.py`` removed and split into other files:

      - common/tf_util.py: ``sample``, ``calc_entropy``, ``mse``, ``avg_norm``, ``total_episode_reward_logger``,
        ``q_explained_variance``, ``gradient_add``, ``avg_norm``, ``check_shape``,
        ``seq_to_batch``, ``batch_to_seq``.
      - common/tf_layers.py: ``conv``, ``linear``, ``lstm``, ``_ln``, ``lnlstm``, ``conv_to_fc``, ``ortho_init``.
      - a2c/a2c.py: ``discount_with_dones``.
      - acer/acer_simple.py: ``get_by_index``, ``EpisodeStats``.
      - common/schedules.py: ``constant``, ``linear_schedule``, ``middle_drop``, ``double_linear_con``, ``double_middle_drop``,
        ``SCHEDULES``, ``Scheduler``.

   - ``trpo_mpi/utils.py`` functions moved (``traj_segment_generator`` moved to ``common/runners.py``, ``flatten_lists`` to ``common/misc_util.py``).
   - ``ppo2/ppo2.py`` functions moved (``safe_mean`` to ``common/math_util.py``, ``constfn`` and ``get_schedule_fn`` to ``common/schedules.py``).
   - ``sac/policies.py`` function ``mlp`` moved to ``common/tf_layers.py``.
   - ``sac/sac.py`` function ``get_vars`` removed (replaced with ``tf.util.get_trainable_vars``).
   - ``deepq/replay_buffer.py`` renamed to ``common/buffers.py``.


New Features:
^^^^^^^^^^^^^
- Parallelized updating and sampling from the replay buffer in DQN. (@flodorner)
- Docker build script, `scripts/build_docker.sh`, can push images automatically.
- Added callback collection
- Added ``unwrap_vec_normalize`` and ``sync_envs_normalization`` in the ``vec_env`` module
  to synchronize two VecNormalize environment
- Added a seeding method for vectorized environments. (@NeoExtended)
- Added extend method to store batches of experience in ReplayBuffer. (@PartiallyTyped)


Bug Fixes:
^^^^^^^^^^

- Fixed Docker images via ``scripts/build_docker.sh`` and ``Dockerfile``: GPU image now contains ``tensorflow-gpu``,
  and both images have ``stable_baselines`` installed in developer mode at correct directory for mounting.
- Fixed Docker GPU run script, ``scripts/run_docker_gpu.sh``, to work with new NVidia Container Toolkit.
- Repeated calls to ``RLModel.learn()`` now preserve internal counters for some episode
  logging statistics that used to be zeroed at the start of every call.
- Fix `DummyVecEnv.render` for ``num_envs > 1``. This used to print a warning and then not render at all. (@shwang)
- Fixed a bug in PPO2, ACER, A2C, and ACKTR where repeated calls to ``learn(total_timesteps)`` reset
  the environment on every call, potentially biasing samples toward early episode timesteps.
  (@shwang)
- Fixed by adding lazy property ``ActorCriticRLModel.runner``. Subclasses now use lazily-generated
    ``self.runner`` instead of reinitializing a new Runner every time ``learn()`` is called.
- Fixed a bug in ``check_env`` where it would fail on high dimensional action spaces
- Fixed ``Monitor.close()`` that was not calling the parent method
- Fixed a bug in ``BaseRLModel`` when seeding vectorized environments. (@NeoExtended)
- Fixed ``num_timesteps`` computation to be consistent between algorithms (updated after ``env.step()``)
  Only ``TRPO`` and ``PPO1`` update it differently (after synchronization) because they rely on MPI
- Fixed bug in ``TRPO`` with NaN standardized advantages (@richardwu)
- Fixed partial minibatch computation in ExpertDataset (@richardwu)
- Fixed normalization (with ``VecNormalize``) for off-policy algorithms
- Fixed ``sync_envs_normalization`` to sync the reward normalization too
- Bump minimum Gym version (>=0.11)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Removed redundant return value from ``a2c.utils::total_episode_reward_logger``. (@shwang)
- Cleanup and refactoring in ``common/identity_env.py`` (@shwang)
- Added a Makefile to simplify common development tasks (build the doc, type check, run the tests)


Documentation:
^^^^^^^^^^^^^^
- Add dedicated page for callbacks
- Fixed example for creating a GIF (@KuKuXia)
- Change Colab links in the README to point to the notebooks repo
- Fix typo in Reinforcement Learning Tips and Tricks page. (@mmcenta)


Release 2.9.0 (2019-12-20)
--------------------------

*Reproducible results, automatic ``VecEnv`` wrapping, env checker and more usability improvements*

Breaking Changes:
^^^^^^^^^^^^^^^^^
- The ``seed`` argument has been moved from `learn()` method to model constructor
  in order to have reproducible results
- ``allow_early_resets`` of the ``Monitor`` wrapper now default to ``True``
- ``make_atari_env`` now returns a ``DummyVecEnv`` by default (instead of a ``SubprocVecEnv``)
  this usually improves performance.
- Fix inconsistency of sample type, so that mode/sample function returns tensor of tf.int64 in CategoricalProbabilityDistribution/MultiCategoricalProbabilityDistribution (@seheevic)

New Features:
^^^^^^^^^^^^^
- Add ``n_cpu_tf_sess`` to model constructor to choose the number of threads used by Tensorflow
- Environments are automatically wrapped in a ``DummyVecEnv`` if needed when passing them to the model constructor
- Added ``stable_baselines.common.make_vec_env`` helper to simplify VecEnv creation
- Added ``stable_baselines.common.evaluation.evaluate_policy`` helper to simplify model evaluation
- ``VecNormalize`` changes:

   - Now supports being pickled and unpickled (@AdamGleave).
   - New methods ``.normalize_obs(obs)`` and `normalize_reward(rews)` apply normalization
     to arbitrary observation or rewards without updating statistics (@shwang)
   - ``.get_original_reward()`` returns the unnormalized rewards from the most recent timestep
   - ``.reset()`` now collects observation statistics (used to only apply normalization)

- Add parameter ``exploration_initial_eps`` to DQN. (@jdossgollin)
- Add type checking and PEP 561 compliance.
  Note: most functions are still not annotated, this will be a gradual process.
- DDPG, TD3 and SAC accept non-symmetric action spaces. (@Antymon)
- Add ``check_env`` util to check if a custom environment follows the gym interface (@araffin and @justinkterry)

Bug Fixes:
^^^^^^^^^^
- Fix seeding, so it is now possible to have deterministic results on cpu
- Fix a bug in DDPG where ``predict`` method with `deterministic=False` would fail
- Fix a bug in TRPO: mean_losses was not initialized causing the logger to crash when there was no gradients (@MarvineGothic)
- Fix a bug in ``cmd_util`` from API change in recent Gym versions
- Fix a bug in DDPG, TD3 and SAC where warmup and random exploration actions would end up scaled in the replay buffer (@Antymon)

Deprecations:
^^^^^^^^^^^^^
- ``nprocs`` (ACKTR) and ``num_procs`` (ACER) are deprecated in favor of ``n_cpu_tf_sess`` which is now common
  to all algorithms
- ``VecNormalize``: ``load_running_average`` and ``save_running_average`` are deprecated in favour of using pickle.

Others:
^^^^^^^
- Add upper bound for Tensorflow version (<2.0.0).
- Refactored test to remove duplicated code
- Add pull request template
- Replaced redundant code in load_results (@jbulow)
- Minor PEP8 fixes in dqn.py (@justinkterry)
- Add a message to the assert in ``PPO2``
- Update replay buffer doctring
- Fix ``VecEnv`` docstrings

Documentation:
^^^^^^^^^^^^^^
- Add plotting to the Monitor example (@rusu24edward)
- Add Snake Game AI project (@pedrohbtp)
- Add note on the support Tensorflow versions.
- Remove unnecessary steps required for Windows installation.
- Remove ``DummyVecEnv`` creation when not needed
- Added ``make_vec_env`` to the examples to simplify VecEnv creation
- Add QuaRL project (@srivatsankrishnan)
- Add Pwnagotchi project (@evilsocket)
- Fix multiprocessing example (@rusu24edward)
- Fix ``result_plotter`` example
- Add JNRR19 tutorial (by @edbeeching, @hill-a and @araffin)
- Updated notebooks link
- Fix typo in algos.rst, "containes" to "contains" (@SyllogismRXS)
- Fix outdated source documentation for load_results
- Add PPO_CPP project (@Antymon)
- Add section on C++ portability of Tensorflow models (@Antymon)
- Update custom env documentation to reflect new gym API for the ``close()`` method (@justinkterry)
- Update custom env documentation to clarify what step and reset return (@justinkterry)
- Add RL tips and tricks for doing RL experiments
- Corrected lots of typos
- Add spell check to documentation if available


Release 2.8.0 (2019-09-29)
--------------------------

**MPI dependency optional, new save format, ACKTR with continuous actions**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- OpenMPI-dependent algorithms (PPO1, TRPO, GAIL, DDPG) are disabled in the
  default installation of stable_baselines. ``mpi4py`` is now installed as an
  extra. When ``mpi4py`` is not available, stable-baselines skips imports of
  OpenMPI-dependent algorithms.
  See :ref:`installation notes <openmpi>` and
  `Issue #430 <https://github.com/hill-a/stable-baselines/issues/430>`_.
- SubprocVecEnv now defaults to a thread-safe start method, ``forkserver`` when
  available and otherwise ``spawn``. This may require application code be
  wrapped in ``if __name__ == '__main__'``. You can restore previous behavior
  by explicitly setting ``start_method = 'fork'``. See
  `PR #428 <https://github.com/hill-a/stable-baselines/pull/428>`_.
- Updated dependencies: tensorflow v1.8.0 is now required
- Removed ``checkpoint_path`` and ``checkpoint_freq`` argument from ``DQN`` that were not used
- Removed ``bench/benchmark.py`` that was not used
- Removed several functions from ``common/tf_util.py`` that were not used
- Removed ``ppo1/run_humanoid.py``

New Features:
^^^^^^^^^^^^^
- **important change** Switch to using zip-archived JSON and Numpy ``savez`` for
  storing models for better support across library/Python versions. (@Miffyli)
- ACKTR now supports continuous actions
- Add ``double_q`` argument to ``DQN`` constructor

Bug Fixes:
^^^^^^^^^^
- Skip automatic imports of OpenMPI-dependent algorithms to avoid an issue
  where OpenMPI would cause stable-baselines to hang on Ubuntu installs.
  See :ref:`installation notes <openmpi>` and
  `Issue #430 <https://github.com/hill-a/stable-baselines/issues/430>`_.
- Fix a bug when calling ``logger.configure()`` with MPI enabled (@keshaviyengar)
- set ``allow_pickle=True`` for numpy>=1.17.0 when loading expert dataset
- Fix a bug when using VecCheckNan with numpy ndarray as state.  `Issue #489 <https://github.com/hill-a/stable-baselines/issues/489>`_. (@ruifeng96150)

Deprecations:
^^^^^^^^^^^^^
- Models saved with cloudpickle format (stable-baselines<=2.7.0) are now
  deprecated in favor of zip-archive format for better support across
  Python/Tensorflow versions. (@Miffyli)

Others:
^^^^^^^
- Implementations of noise classes (``AdaptiveParamNoiseSpec``, ``NormalActionNoise``,
  ``OrnsteinUhlenbeckActionNoise``) were moved from `stable_baselines.ddpg.noise`
  to ``stable_baselines.common.noise``. The API remains backward-compatible;
  for example ``from stable_baselines.ddpg.noise import NormalActionNoise`` is still
  okay. (@shwang)
- Docker images were updated
- Cleaned up files in ``common/`` folder and in `acktr/` folder that were only used by old ACKTR version
  (e.g. `filter.py`)
- Renamed `acktr_disc.py` to `acktr.py`

Documentation:
^^^^^^^^^^^^^^
- Add WaveRL project (@jaberkow)
- Add Fenics-DRL project (@DonsetPG)
- Fix and rename custom policy names (@eavelardev)
- Add documentation on exporting models.
- Update maintainers list (Welcome to @Miffyli)


Release 2.7.0 (2019-07-31)
--------------------------

**Twin Delayed DDPG (TD3) and GAE bug fix (TRPO, PPO1, GAIL)**

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- added Twin Delayed DDPG (TD3) algorithm, with HER support
- added support for continuous action spaces to ``action_probability``, computing the PDF of a Gaussian
  policy in addition to the existing support for categorical stochastic policies.
- added flag to ``action_probability`` to return log-probabilities.
- added support for python lists and numpy arrays in ``logger.writekvs``. (@dwiel)
- the info dict returned by VecEnvs now include a ``terminal_observation`` key providing access to the last observation in a trajectory. (@qxcv)

Bug Fixes:
^^^^^^^^^^
- fixed a bug in ``traj_segment_generator`` where the ``episode_starts`` was wrongly recorded,
  resulting in wrong calculation of Generalized Advantage Estimation (GAE), this affects TRPO, PPO1 and GAIL (thanks to @miguelrass for spotting the bug)
- added missing property ``n_batch`` in ``BasePolicy``.

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- renamed some keys in ``traj_segment_generator`` to be more meaningful
- retrieve unnormalized reward when using Monitor wrapper with TRPO, PPO1 and GAIL
  to display them in the logs (mean episode reward)
- clean up DDPG code (renamed variables)

Documentation:
^^^^^^^^^^^^^^

- doc fix for the hyperparameter tuning command in the rl zoo
- added an example on how to log additional variable with tensorboard and a callback



Release 2.6.0 (2019-06-12)
--------------------------

**Hindsight Experience Replay (HER) - Reloaded | get/load parameters**

Breaking Changes:
^^^^^^^^^^^^^^^^^

- **breaking change** removed ``stable_baselines.ddpg.memory`` in favor of ``stable_baselines.deepq.replay_buffer`` (see fix below)

**Breaking Change:** DDPG replay buffer was unified with DQN/SAC replay buffer. As a result,
when loading a DDPG model trained with stable_baselines<2.6.0, it throws an import error.
You can fix that using:

.. code-block:: python

  import sys
  import pkg_resources

  import stable_baselines

  # Fix for breaking change for DDPG buffer in v2.6.0
  if pkg_resources.get_distribution("stable_baselines").version >= "2.6.0":
      sys.modules['stable_baselines.ddpg.memory'] = stable_baselines.deepq.replay_buffer
      stable_baselines.deepq.replay_buffer.Memory = stable_baselines.deepq.replay_buffer.ReplayBuffer


We recommend you to save again the model afterward, so the fix won't be needed the next time the trained agent is loaded.


New Features:
^^^^^^^^^^^^^

- **revamped HER implementation**: clean re-implementation from scratch, now supports DQN, SAC and DDPG
- add ``action_noise`` param for SAC, it helps exploration for problem with deceptive reward
- The parameter ``filter_size`` of the function ``conv`` in A2C utils now supports passing a list/tuple of two integers (height and width), in order to have non-squared kernel matrix. (@yutingsz)
- add ``random_exploration`` parameter for DDPG and SAC, it may be useful when using HER + DDPG/SAC. This hack was present in the original OpenAI Baselines DDPG + HER implementation.
- added ``load_parameters`` and ``get_parameters`` to base RL class. With these methods, users are able to load and get parameters to/from existing model, without touching tensorflow. (@Miffyli)
- added specific hyperparameter for PPO2 to clip the value function (``cliprange_vf``)
- added ``VecCheckNan`` wrapper

Bug Fixes:
^^^^^^^^^^

- bugfix for ``VecEnvWrapper.__getattr__`` which enables access to class attributes inherited from parent classes.
- fixed path splitting in ``TensorboardWriter._get_latest_run_id()`` on Windows machines (@PatrickWalter214)
- fixed a bug where initial learning rate is logged instead of its placeholder in ``A2C.setup_model`` (@sc420)
- fixed a bug where number of timesteps is incorrectly updated and logged in ``A2C.learn`` and ``A2C._train_step`` (@sc420)
- fixed ``num_timesteps`` (total_timesteps) variable in PPO2 that was wrongly computed.
- fixed a bug in DDPG/DQN/SAC, when there were the number of samples in the replay buffer was lesser than the batch size
  (thanks to @dwiel for spotting the bug)
- **removed** ``a2c.utils.find_trainable_params`` please use ``common.tf_util.get_trainable_vars`` instead.
  ``find_trainable_params`` was returning all trainable variables, discarding the scope argument.
  This bug was causing the model to save duplicated parameters (for DDPG and SAC)
  but did not affect the performance.

Deprecations:
^^^^^^^^^^^^^

- **deprecated** ``memory_limit`` and ``memory_policy`` in DDPG, please use ``buffer_size`` instead. (will be removed in v3.x.x)

Others:
^^^^^^^

- **important change** switched to using dictionaries rather than lists when storing parameters, with tensorflow Variable names being the keys. (@Miffyli)
- removed unused dependencies (tdqm, dill, progressbar2, seaborn, glob2, click)
- removed ``get_available_gpus`` function which hadn't been used anywhere (@Pastafarianist)

Documentation:
^^^^^^^^^^^^^^

- added guide for managing ``NaN`` and ``inf``
- updated ven_env doc
- misc doc updates

Release 2.5.1 (2019-05-04)
--------------------------

**Bug fixes + improvements in the VecEnv**

**Warning: breaking changes when using custom policies**

- doc update (fix example of result plotter + improve doc)
- fixed logger issues when stdout lacks ``read`` function
- fixed a bug in ``common.dataset.Dataset`` where shuffling was not disabled properly (it affects only PPO1 with recurrent policies)
- fixed output layer name for DDPG q function, used in pop-art normalization and l2 regularization of the critic
- added support for multi env recording to ``generate_expert_traj`` (@XMaster96)
- added support for LSTM model recording to ``generate_expert_traj`` (@XMaster96)
- ``GAIL``: remove mandatory matplotlib dependency and refactor as subclass of ``TRPO`` (@kantneel and @AdamGleave)
- added ``get_attr()``, ``env_method()`` and ``set_attr()`` methods for all VecEnv.
  Those methods now all accept ``indices`` keyword to select a subset of envs.
  ``set_attr`` now returns ``None`` rather than a list of ``None``. (@kantneel)
- ``GAIL``: ``gail.dataset.ExpertDataset`` supports loading from memory rather than file, and
  ``gail.dataset.record_expert`` supports returning in-memory rather than saving to file.
- added support in ``VecEnvWrapper`` for accessing attributes of arbitrarily deeply nested
  instances of ``VecEnvWrapper`` and ``VecEnv``. This is allowed as long as the attribute belongs
  to exactly one of the nested instances i.e. it must be unambiguous. (@kantneel)
- fixed bug where result plotter would crash on very short runs (@Pastafarianist)
- added option to not trim output of result plotter by number of timesteps (@Pastafarianist)
- clarified the public interface of ``BasePolicy`` and ``ActorCriticPolicy``. **Breaking change** when using custom policies: ``masks_ph`` is now called ``dones_ph``,
  and most placeholders were made private: e.g. ``self.value_fn`` is now ``self._value_fn``
- support for custom stateful policies.
- fixed episode length recording in ``trpo_mpi.utils.traj_segment_generator`` (@GerardMaggiolino)


Release 2.5.0 (2019-03-28)
--------------------------

**Working GAIL, pretrain RL models and hotfix for A2C with continuous actions**

- fixed various bugs in GAIL
- added scripts to generate dataset for gail
- added tests for GAIL + data for Pendulum-v0
- removed unused ``utils`` file in DQN folder
- fixed a bug in A2C where actions were cast to ``int32`` even in the continuous case
- added addional logging to A2C when Monitor wrapper is used
- changed logging for PPO2: do not display NaN when reward info is not present
- change default value of A2C lr schedule
- removed behavior cloning script
- added ``pretrain`` method to base class, in order to use behavior cloning on all models
- fixed ``close()`` method for DummyVecEnv.
- added support for Dict spaces in DummyVecEnv and SubprocVecEnv. (@AdamGleave)
- added support for arbitrary multiprocessing start methods and added a warning about SubprocVecEnv that are not thread-safe by default.  (@AdamGleave)
- added support for Discrete actions for GAIL
- fixed deprecation warning for tf: replaces ``tf.to_float()`` by ``tf.cast()``
- fixed bug in saving and loading ddpg model when using normalization of obs or returns (@tperol)
- changed DDPG default buffer size from 100 to 50000.
- fixed a bug in ``ddpg.py`` in ``combined_stats`` for eval. Computed mean on ``eval_episode_rewards`` and ``eval_qs`` (@keshaviyengar)
- fixed a bug in ``setup.py`` that would error on non-GPU systems without TensorFlow installed


Release 2.4.1 (2019-02-11)
--------------------------

**Bug fixes and improvements**

- fixed computation of training metrics in TRPO and PPO1
- added ``reset_num_timesteps`` keyword when calling train() to continue tensorboard learning curves
- reduced the size taken by tensorboard logs (added a ``full_tensorboard_log`` to enable full logging, which was the previous behavior)
- fixed image detection for tensorboard logging
- fixed ACKTR for recurrent policies
- fixed gym breaking changes
- fixed custom policy examples in the doc for DQN and DDPG
- remove gym spaces patch for equality functions
- fixed tensorflow dependency: cpu version was installed overwritting tensorflow-gpu when present.
- fixed a bug in ``traj_segment_generator`` (used in ppo1 and trpo) where ``new`` was not updated. (spotted by @junhyeokahn)


Release 2.4.0 (2019-01-17)
--------------------------

**Soft Actor-Critic (SAC) and policy kwargs**

- added Soft Actor-Critic (SAC) model
- fixed a bug in DQN where prioritized_replay_beta_iters param was not used
- fixed DDPG that did not save target network parameters
- fixed bug related to shape of true_reward (@abhiskk)
- fixed example code in documentation of tf_util:Function (@JohannesAck)
- added learning rate schedule for SAC
- fixed action probability for continuous actions with actor-critic models
- added optional parameter to action_probability for likelihood calculation of given action being taken.
- added more flexible custom LSTM policies
- added auto entropy coefficient optimization for SAC
- clip continuous actions at test time too for all algorithms (except SAC/DDPG where it is not needed)
- added a mean to pass kwargs to policy when creating a model (+ save those kwargs)
- fixed DQN examples in DQN folder
- added possibility to pass activation function for DDPG, DQN and SAC


Release 2.3.0 (2018-12-05)
--------------------------

- added support for storing model in file like object. (thanks to @ernestum)
- fixed wrong image detection when using tensorboard logging with DQN
- fixed bug in ppo2 when passing non callable lr after loading
- fixed tensorboard logging in ppo2 when nminibatches=1
- added early stoppping via callback return value (@ernestum)
- added more flexible custom mlp policies (@ernestum)


Release 2.2.1 (2018-11-18)
--------------------------

- added VecVideoRecorder to record mp4 videos from environment.


Release 2.2.0 (2018-11-07)
--------------------------

- Hotfix for ppo2, the wrong placeholder was used for the value function


Release 2.1.2 (2018-11-06)
--------------------------

- added ``async_eigen_decomp`` parameter for ACKTR and set it to ``False`` by default (remove deprecation warnings)
- added methods for calling env methods/setting attributes inside a VecEnv (thanks to @bjmuld)
- updated gym minimum version


Release 2.1.1 (2018-10-20)
--------------------------

- fixed MpiAdam synchronization issue in PPO1 (thanks to @brendenpetersen) issue #50
- fixed dependency issues (new mujoco-py requires a mujoco license + gym broke MultiDiscrete space shape)


Release 2.1.0 (2018-10-2)
-------------------------

.. warning::

	This version contains breaking changes for DQN policies, please read the full details

**Bug fixes + doc update**


- added patch fix for equal function using `gym.spaces.MultiDiscrete` and `gym.spaces.MultiBinary`
- fixes for DQN action_probability
- re-added double DQN + refactored DQN policies **breaking changes**
- replaced ``async`` with ``async_eigen_decomp`` in ACKTR/KFAC for python 3.7 compatibility
- removed action clipping for prediction of continuous actions (see issue #36)
- fixed NaN issue due to clipping the continuous action in the wrong place (issue #36)
- documentation was updated (policy + DDPG example hyperparameters)

Release 2.0.0 (2018-09-18)
--------------------------

.. warning::

	This version contains breaking changes, please read the full details

**Tensorboard, refactoring and bug fixes**


- Renamed DeepQ to DQN **breaking changes**
- Renamed DeepQPolicy to DQNPolicy **breaking changes**
- fixed DDPG behavior **breaking changes**
- changed default policies for DDPG, so that DDPG now works correctly **breaking changes**
- added more documentation (some modules from common).
- added doc about using custom env
- added Tensorboard support for A2C, ACER, ACKTR, DDPG, DeepQ, PPO1, PPO2 and TRPO
- added episode reward to Tensorboard
- added documentation for Tensorboard usage
- added Identity for Box action space
- fixed render function ignoring parameters when using wrapped environments
- fixed PPO1 and TRPO done values for recurrent policies
- fixed image normalization not occurring when using images
- updated VecEnv objects for the new Gym version
- added test for DDPG
- refactored DQN policies
- added registry for policies, can be passed as string to the agent
- added documentation for custom policies + policy registration
- fixed numpy warning when using DDPG Memory
- fixed DummyVecEnv not copying the observation array when stepping and resetting
- added pre-built docker images + installation instructions
- added ``deterministic`` argument in the predict function
- added assert in PPO2 for recurrent policies
- fixed predict function to handle both vectorized and unwrapped environment
- added input check to the predict function
- refactored ActorCritic models to reduce code duplication
- refactored Off Policy models (to begin HER and replay_buffer refactoring)
- added tests for auto vectorization detection
- fixed render function, to handle positional arguments


Release 1.0.7 (2018-08-29)
--------------------------

**Bug fixes and documentation**

- added html documentation using sphinx + integration with read the docs
- cleaned up README + typos
- fixed normalization for DQN with images
- fixed DQN identity test


Release 1.0.1 (2018-08-20)
--------------------------

**Refactored Stable Baselines**

- refactored A2C, ACER, ACTKR, DDPG, DeepQ, GAIL, TRPO, PPO1 and PPO2 under a single constant class
- added callback to refactored algorithm training
- added saving and loading to refactored algorithms
- refactored ACER, DDPG, GAIL, PPO1 and TRPO to fit with A2C, PPO2 and ACKTR policies
- added new policies for most algorithms (Mlp, MlpLstm, MlpLnLstm, Cnn, CnnLstm and CnnLnLstm)
- added dynamic environment switching (so continual RL learning is now feasible)
- added prediction from observation and action probability from observation for all the algorithms
- fixed graphs issues, so models wont collide in names
- fixed behavior_clone weight loading for GAIL
- fixed Tensorflow using all the GPU VRAM
- fixed models so that they are all compatible with vectorized environments
- fixed ``set_global_seed`` to update ``gym.spaces``'s random seed
- fixed PPO1 and TRPO performance issues when learning identity function
- added new tests for loading, saving, continuous actions and learning the identity function
- fixed DQN wrapping for atari
- added saving and loading for Vecnormalize wrapper
- added automatic detection of action space (for the policy network)
- fixed ACER buffer with constant values assuming n_stack=4
- fixed some RL algorithms not clipping the action to be in the action_space, when using ``gym.spaces.Box``
- refactored algorithms can take either a ``gym.Environment`` or a ``str`` ([if the environment name is registered](https://github.com/openai/gym/wiki/Environments))
- Hoftix in ACER (compared to v1.0.0)

Future Work :

- Finish refactoring HER
- Refactor ACKTR and ACER for continuous implementation



Release 0.1.6 (2018-07-27)
--------------------------

**Deobfuscation of the code base + pep8 and fixes**

-  Fixed ``tf.session().__enter__()`` being used, rather than
   ``sess = tf.session()`` and passing the session to the objects
-  Fixed uneven scoping of TensorFlow Sessions throughout the code
-  Fixed rolling vecwrapper to handle observations that are not only
   grayscale images
-  Fixed deepq saving the environment when trying to save itself
-  Fixed
   ``ValueError: Cannot take the length of Shape with unknown rank.`` in
   ``acktr``, when running ``run_atari.py`` script.
-  Fixed calling baselines sequentially no longer creates graph
   conflicts
-  Fixed mean on empty array warning with deepq
-  Fixed kfac eigen decomposition not cast to float64, when the
   parameter use_float64 is set to True
-  Fixed Dataset data loader, not correctly resetting id position if
   shuffling is disabled
-  Fixed ``EOFError`` when reading from connection in the ``worker`` in
   ``subproc_vec_env.py``
-  Fixed ``behavior_clone`` weight loading and saving for GAIL
-  Avoid taking root square of negative number in ``trpo_mpi.py``
-  Removed some duplicated code (a2cpolicy, trpo_mpi)
-  Removed unused, undocumented and crashing function ``reset_task`` in
   ``subproc_vec_env.py``
-  Reformated code to PEP8 style
-  Documented all the codebase
-  Added atari tests
-  Added logger tests

Missing: tests for acktr continuous (+ HER, rely on mujoco...)

Maintainers
-----------

Stable-Baselines is currently maintained by `Ashley Hill`_ (aka @hill-a), `Antonin Raffin`_ (aka `@araffin`_),
`Maximilian Ernestus`_ (aka @ernestum), `Adam Gleave`_ (`@AdamGleave`_) and `Anssi Kanervisto`_ (aka `@Miffyli`_).

.. _Ashley Hill: https://github.com/hill-a
.. _Antonin Raffin: https://araffin.github.io/
.. _Maximilian Ernestus: https://github.com/ernestum
.. _Adam Gleave: https://gleave.me/
.. _@araffin: https://github.com/araffin
.. _@AdamGleave: https://github.com/adamgleave
.. _Anssi Kanervisto: https://github.com/Miffyli
.. _@Miffyli: https://github.com/Miffyli


Contributors (since v2.0.0):
----------------------------
In random order...

Thanks to @bjmuld @iambenzo @iandanforth @r7vme @brendenpetersen @huvar @abhiskk @JohannesAck @mily20001
@EliasHasle @mrakgr @Bleyddyn @antoine-galataud @junhyeokahn @AdamGleave @keshaviyengar @tperol
@XMaster96 @kantneel @Pastafarianist @GerardMaggiolino @PatrickWalter214 @yutingsz @sc420 @Aaahh @billtubbs
@Miffyli @dwiel @miguelrass @qxcv @jaberkow @eavelardev @ruifeng96150 @pedrohbtp @srivatsankrishnan @evilsocket
@MarvineGothic @jdossgollin @SyllogismRXS @rusu24edward @jbulow @Antymon @seheevic @justinkterry @edbeeching
@flodorner @KuKuXia @NeoExtended @PartiallyTyped @mmcenta @richardwu @tirafesi @caburu @johannes-dornheim @kvenkman @aakash94
@enderdead @hardmaru @jbarsce @ColinLeongUDRI @shwang @YangRui2015 @sophiagu @OGordon100 @SVJayanthi @sunshineclt
@roccivic @anj1