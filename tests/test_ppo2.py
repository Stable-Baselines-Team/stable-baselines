import os

import pytest
import gym

from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv


@pytest.mark.parametrize("cliprange", [0.2, lambda x: 0.1 * x])
@pytest.mark.parametrize("cliprange_vf", [None, 0.2, lambda x: 0.3 * x, -1.0])
def test_clipping(tmp_path, cliprange, cliprange_vf):
    """Test the different clipping (policy and vf)"""
    model = PPO2(
        "MlpPolicy",
        "CartPole-v1",
        cliprange=cliprange,
        cliprange_vf=cliprange_vf,
        noptepochs=2,
        n_steps=64,
    ).learn(100)
    save_path = os.path.join(str(tmp_path), "ppo2_clip.zip")
    model.save(save_path)
    env = model.get_env()
    model = PPO2.load(save_path, env=env)
    model.learn(100)

    if os.path.exists(save_path):
        os.remove(save_path)


def test_ppo2_update_n_batch_on_load(tmp_path):
    env = make_vec_env("CartPole-v1", n_envs=2)
    model = PPO2("MlpPolicy", env, n_steps=10, nminibatches=1)
    save_path = os.path.join(str(tmp_path), "ppo2_cartpole.zip")

    model.learn(total_timesteps=100)
    model.save(save_path)

    del model

    model = PPO2.load(save_path)
    test_env = DummyVecEnv([lambda: gym.make("CartPole-v1")])

    model.set_env(test_env)
    model.learn(total_timesteps=100)
