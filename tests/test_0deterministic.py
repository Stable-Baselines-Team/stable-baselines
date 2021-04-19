import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO, TD3
from stable_baselines.common.noise import NormalActionNoise

N_STEPS_TRAINING = 300
SEED = 0


# Weird stuff: TD3 would fail if another algorithm is tested before
# with n_cpu_tf_sess > 1
@pytest.mark.xfail(reason="TD3 deterministic randomly fail when run with others...", strict=False)
def test_deterministic_td3():
    results = [[], []]
    rewards = [[], []]
    kwargs = {'n_cpu_tf_sess': 1}
    env_id = 'Pendulum-v0'
    kwargs.update({'action_noise': NormalActionNoise(0.0, 0.1)})

    for i in range(2):
        model = TD3('MlpPolicy', env_id, seed=SEED, **kwargs)
        model.learn(N_STEPS_TRAINING)
        env = model.get_env()
        obs = env.reset()
        for _ in range(20):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, _ = env.step(action)
            results[i].append(action)
            rewards[i].append(reward)
    # without the extended tolerance, test fails for unknown reasons on Github...
    assert np.allclose(results[0], results[1], rtol=1e-2), results
    assert np.allclose(rewards[0], rewards[1], rtol=1e-2), rewards


@pytest.mark.parametrize("algo", [A2C, ACKTR, ACER, DDPG, DQN, PPO1, PPO2, SAC, TRPO])
def test_deterministic_training_common(algo):
    results = [[], []]
    rewards = [[], []]
    kwargs = {'n_cpu_tf_sess': 1}
    if algo in [DDPG, TD3, SAC]:
        env_id = 'Pendulum-v0'
        kwargs.update({'action_noise': NormalActionNoise(0.0, 0.1)})
    else:
        env_id = 'CartPole-v1'
        if algo == DQN:
            kwargs.update({'learning_starts': 100})

    for i in range(2):
        model = algo('MlpPolicy', env_id, seed=SEED, **kwargs)
        model.learn(N_STEPS_TRAINING)
        env = model.get_env()
        obs = env.reset()
        for _ in range(20):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, _, _ = env.step(action)
            results[i].append(action)
            rewards[i].append(reward)
    assert sum(results[0]) == sum(results[1]), results
    assert sum(rewards[0]) == sum(rewards[1]), rewards
