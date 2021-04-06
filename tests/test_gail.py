import os
import shutil

import gym
import numpy as np
import pytest

from stable_baselines import (A2C, ACER, ACKTR, GAIL, DDPG, DQN, PPO1, PPO2,
                              TD3, TRPO, SAC)
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import CheckpointCallback
from stable_baselines.gail import ExpertDataset, generate_expert_traj


EXPERT_PATH_PENDULUM = "stable_baselines/gail/dataset/expert_pendulum.npz"
EXPERT_PATH_DISCRETE = "stable_baselines/gail/dataset/expert_cartpole.npz"


@pytest.mark.parametrize("expert_env", [('Pendulum-v0', EXPERT_PATH_PENDULUM, True),
                                        ('CartPole-v1', EXPERT_PATH_DISCRETE, False)])
def test_gail(tmp_path, expert_env):
    env_id, expert_path, load_from_memory = expert_env
    env = gym.make(env_id)

    traj_data = None
    if load_from_memory:
        traj_data = np.load(expert_path)
        expert_path = None
    dataset = ExpertDataset(traj_data=traj_data, expert_path=expert_path, traj_limitation=10,
                            sequential_preprocessing=True)

    # Note: train for 1M steps to have a working policy
    model = GAIL('MlpPolicy', env, adversary_entcoeff=0.0, lam=0.92, max_kl=0.001,
                 expert_dataset=dataset, hidden_size_adversary=64, verbose=0)

    model.learn(300)
    model.save(str(tmp_path / "GAIL-{}".format(env_id)))
    model = model.load(str(tmp_path / "GAIL-{}".format(env_id)), env=env)
    model.learn(300)

    evaluate_policy(model, env, n_eval_episodes=5)
    del dataset, model


@pytest.mark.parametrize("generate_env", [
                                            (SAC, 'MlpPolicy', 'Pendulum-v0', 1, 10),
                                            (DQN, 'MlpPolicy', 'CartPole-v1', 1, 10),
                                            (A2C, 'MlpLstmPolicy', 'Pendulum-v0', 1, 10),
                                            (A2C, 'MlpLstmPolicy', 'CartPole-v1', 1, 10),
                                            (A2C, 'CnnPolicy', 'BreakoutNoFrameskip-v4', 8, 1),
                                          ])
def test_generate(tmp_path, generate_env):
    model, policy, env_name, n_env, n_episodes = generate_env

    if n_env > 1:
        env = make_atari_env(env_name, num_env=n_env, seed=0)
        model = model(policy, env, verbose=0)
    else:
        model = model(policy, env_name, verbose=0)

    dataset = generate_expert_traj(model, str(tmp_path / 'expert'), n_timesteps=300, n_episodes=n_episodes,
                                   image_folder=str(tmp_path / 'test_recorded_images'))

    assert set(dataset.keys()).issuperset(['actions', 'obs', 'rewards', 'episode_returns', 'episode_starts'])
    assert sum(dataset['episode_starts']) == n_episodes
    assert len(dataset['episode_returns']) == n_episodes
    n_timesteps = len(dataset['episode_starts'])
    for key, val in dataset.items():
        if key != 'episode_returns':
            assert val.shape[0] == n_timesteps, "inconsistent number of timesteps at '{}'".format(key)

    dataset_loaded = np.load(str(tmp_path / 'expert.npz'), allow_pickle=True)
    assert dataset.keys() == dataset_loaded.keys()
    for key in dataset.keys():
        assert (dataset[key] == dataset_loaded[key]).all(), "different data at '{}'".format(key)
    # Cleanup folder
    if os.path.isdir(str(tmp_path / 'test_recorded_images')):
        shutil.rmtree(str(tmp_path / 'test_recorded_images'))


def test_generate_callable(tmp_path):
    """
    Test generating expert trajectories with a callable.
    """
    env = gym.make("CartPole-v1")
    # Here the expert is a random agent
    def dummy_expert(_obs):
        return env.action_space.sample()
    generate_expert_traj(dummy_expert, tmp_path / 'dummy_expert_cartpole', env, n_timesteps=0, n_episodes=10)


@pytest.mark.xfail(reason="Not Enough Memory", strict=False)
def test_pretrain_images(tmp_path):
    env = make_atari_env("PongNoFrameskip-v4", num_env=1, seed=0)
    env = VecFrameStack(env, n_stack=3)
    model = PPO2('CnnPolicy', env)
    generate_expert_traj(model, str(tmp_path / 'expert_pong'), n_timesteps=0, n_episodes=1,
                         image_folder=str(tmp_path / 'pretrain_recorded_images'))

    expert_path = str(tmp_path / 'expert_pong.npz')
    dataset = ExpertDataset(expert_path=expert_path, traj_limitation=1, batch_size=32,
                            sequential_preprocessing=True)
    model.pretrain(dataset, n_epochs=2)

    shutil.rmtree(str(tmp_path / 'pretrain_recorded_images'))
    env.close()
    del dataset, model, env


def test_gail_callback(tmp_path):
    dataset = ExpertDataset(expert_path=EXPERT_PATH_PENDULUM, traj_limitation=10,
                            sequential_preprocessing=True, verbose=0)
    model = GAIL("MlpPolicy", "Pendulum-v0", dataset)
    checkpoint_callback = CheckpointCallback(save_freq=150, save_path=str(tmp_path / 'logs/gail/'), name_prefix='gail')
    model.learn(total_timesteps=301, callback=checkpoint_callback)
    shutil.rmtree(str(tmp_path / 'logs/gail/'))
    del dataset, model


@pytest.mark.parametrize("model_class", [A2C, ACKTR, GAIL, DDPG, PPO1, PPO2, SAC, TD3, TRPO])
def test_behavior_cloning_box(tmp_path, model_class):
    """
    Behavior cloning with continuous actions.
    """
    dataset = ExpertDataset(expert_path=EXPERT_PATH_PENDULUM, traj_limitation=10,
                            sequential_preprocessing=True, verbose=0)
    model = model_class("MlpPolicy", "Pendulum-v0")
    model.pretrain(dataset, n_epochs=5)
    model.save(str(tmp_path / "test-pretrain"))
    del dataset, model


@pytest.mark.parametrize("model_class", [A2C, ACER, ACKTR, DQN, GAIL, PPO1, PPO2, TRPO])
def test_behavior_cloning_discrete(tmp_path, model_class):
    dataset = ExpertDataset(expert_path=EXPERT_PATH_DISCRETE, traj_limitation=10,
                            sequential_preprocessing=True, verbose=0)
    model = model_class("MlpPolicy", "CartPole-v1")
    model.pretrain(dataset, n_epochs=5)
    model.save(str(tmp_path / "test-pretrain"))
    del dataset, model


def test_dataset_param_validation():
    with pytest.raises(ValueError):
        ExpertDataset()

    traj_data = np.load(EXPERT_PATH_PENDULUM)
    with pytest.raises(ValueError):
        ExpertDataset(traj_data=traj_data, expert_path=EXPERT_PATH_PENDULUM)


def test_generate_vec_env_non_image_observation():
    env = DummyVecEnv([lambda: gym.make('CartPole-v1')] * 2)

    model = PPO2('MlpPolicy', env)
    model.learn(total_timesteps=300)

    generate_expert_traj(model, save_path='.', n_timesteps=0, n_episodes=5)
