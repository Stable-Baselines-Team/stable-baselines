import pytest
import numpy as np

from stable_baselines import A2C, ACER, ACKTR, DQN, DDPG, SAC, PPO1, PPO2, TD3, TRPO
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy


# Hyperparameters for learning identity for each RL model
LEARN_FUNC_DICT = {
    "a2c": lambda e: A2C(
        policy="MlpPolicy",
        learning_rate=1e-3,
        n_steps=4,
        gamma=0.4,
        ent_coef=0.0,
        env=e,
        seed=0,
    ).learn(total_timesteps=4000),
    "acer": lambda e: ACER(
        policy="MlpPolicy",
        env=e,
        seed=0,
        n_steps=4,
        replay_ratio=1,
        ent_coef=0.0,
    ).learn(total_timesteps=4000),
    "acktr": lambda e: ACKTR(
        policy="MlpPolicy", env=e, seed=0, learning_rate=5e-4, ent_coef=0.0, n_steps=4
    ).learn(total_timesteps=4000),
    "dqn": lambda e: DQN(
        policy="MlpPolicy",
        batch_size=32,
        gamma=0.1,
        learning_starts=0,
        exploration_final_eps=0.05,
        exploration_fraction=0.1,
        env=e,
        seed=0,
    ).learn(total_timesteps=4000),
    "ppo1": lambda e: PPO1(
        policy="MlpPolicy",
        env=e,
        seed=0,
        lam=0.5,
        entcoeff=0.0,
        optim_batchsize=16,
        gamma=0.4,
        optim_stepsize=1e-3,
    ).learn(total_timesteps=3000),
    "ppo2": lambda e: PPO2(
        policy="MlpPolicy",
        env=e,
        seed=0,
        learning_rate=1.5e-3,
        lam=0.8,
        ent_coef=0.0,
        gamma=0.4,
    ).learn(total_timesteps=3000),
    "trpo": lambda e: TRPO(
        policy="MlpPolicy",
        env=e,
        gamma=0.4,
        seed=0,
        max_kl=0.05,
        lam=0.7,
        timesteps_per_batch=256,
    ).learn(total_timesteps=4000),
}


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name", ["a2c", "acer", "acktr", "dqn", "ppo1", "ppo2", "trpo"]
)
def test_identity_discrete(model_name):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)

    :param model_name: (str) Name of the RL model
    """
    env = DummyVecEnv([lambda: IdentityEnv(10)])

    model = LEARN_FUNC_DICT[model_name](env)
    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)

    obs = env.reset()
    assert model.action_probability(obs).shape == (
        1,
        10,
    ), "Error: action_probability not returning correct shape"
    action = env.action_space.sample()
    action_prob = model.action_probability(obs, actions=action)
    assert np.prod(action_prob.shape) == 1, "Error: not scalar probability"
    action_logprob = model.action_probability(obs, actions=action, logp=True)
    assert np.allclose(action_prob, np.exp(action_logprob)), (
        action_prob,
        action_logprob,
    )

    # Free memory
    del model, env


@pytest.mark.slow
@pytest.mark.parametrize("model_class", [DDPG, TD3, SAC])
def test_identity_continuous(model_class):
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    n_steps = {SAC: 700, TD3: 500, DDPG: 2000}[model_class]

    kwargs = dict(seed=0, gamma=0.95, buffer_size=1e5)
    if model_class in [DDPG, TD3]:
        n_actions = 1
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions)
        )
        kwargs["action_noise"] = action_noise

    if model_class == DDPG:
        kwargs["actor_lr"] = 1e-3
        kwargs["batch_size"] = 100

    model = model_class("MlpPolicy", env, **kwargs)
    model.learn(total_timesteps=n_steps)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90)
    # Free memory
    del model, env
