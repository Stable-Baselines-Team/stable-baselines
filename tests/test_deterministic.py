import pytest

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TRPO
from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines.common import set_global_seeds
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv

PARAM_NOISE_DDPG = AdaptiveParamNoiseSpec(initial_stddev=float(0.2), desired_action_stddev=float(0.2))

# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    'a2c': lambda e: A2C(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'acer': lambda e: ACER(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'acktr': lambda e: ACKTR(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'dqn': lambda e: DQN(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'ddpg': lambda e: DDPG(policy="MlpPolicy", env=e, param_noise=PARAM_NOISE_DDPG).learn(total_timesteps=1000),
    'ppo1': lambda e: PPO1(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'ppo2': lambda e: PPO2(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'sac': lambda e: SAC(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
    'trpo': lambda e: TRPO(policy="MlpPolicy", env=e).learn(total_timesteps=1000),
}

N_STEPS_TRAINING = 1000
SEED = 0

# NOTE: not working with ACKTR yet
@pytest.mark.parametrize("algo", [A2C, ACER, PPO1, PPO2, TRPO])
def test_deterministic_training_common(algo):
    results = [[], []]
    rewards = [[], []]
    for i in range(2):
        set_global_seeds(0)
        model = algo('MlpPolicy', 'CartPole-v1')
        if hasattr(model.env, 'seed'):
            model.env.seed(SEED)
        else:
            model.env.env_method("seed", SEED)
        model.learn(N_STEPS_TRAINING)
        env = model.get_env()
        obs = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, _, _ = env.step(action)
            results[i].append(action)
            rewards[i].append(reward)
    assert sum(results[0]) == sum(results[1]), print(results)
    assert sum(rewards[0]) == sum(rewards[1]), print(rewards)


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'acer', 'acktr', 'dqn', 'ppo1', 'ppo2', 'trpo'])
def test_identity(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = LEARN_FUNC_DICT[model_name](env)

    n_trials = 1000
    obs = env.reset()
    action_shape = model.predict(obs, deterministic=False)[0].shape
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == action_shape
    for _ in range(n_trials):
        new_action = model.predict(obs, deterministic=True)[0]
        assert action == model.predict(obs, deterministic=True)[0]
        assert new_action.shape == action_shape
    # Free memory
    del model, env


@pytest.mark.slow
@pytest.mark.parametrize("model_name", ['a2c', 'ddpg', 'ppo1', 'ppo2', 'sac', 'trpo'])
def test_identity_continuous(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    model = LEARN_FUNC_DICT[model_name](env)

    n_trials = 1000
    obs = env.reset()
    action_shape = model.predict(obs, deterministic=False)[0].shape
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == action_shape
    for _ in range(n_trials):
        new_action = model.predict(obs, deterministic=True)[0]
        assert action == model.predict(obs, deterministic=True)[0]
        assert new_action.shape == action_shape
