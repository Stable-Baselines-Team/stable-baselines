from gym import spaces
import numpy as np

from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper


class HERWrapper(VecEnvWrapper):
    def __init__(self, venv, reward_function):
        if isinstance(venv.observation_space, spaces.Discrete):
            observation_space = spaces.MultiDiscrete([venv.observation_space.n] * 2)
        elif isinstance(venv.observation_space, spaces.Box):
            low = venv.observation_space.low
            if hasattr(low, "__iter__"):
                low = list(low) * 2

            high = venv.observation_space.low
            if hasattr(high, "__iter__"):
                high = list(high) * 2

            shape = venv.observation_space.shape
            shape[-1] = shape[-1] * 2
            observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=venv.observation_space.dtype)
        elif isinstance(venv.observation_space, spaces.MultiDiscrete):
            observation_space = spaces.MultiDiscrete(venv.observation_space.nvec * 2)
        elif isinstance(venv.observation_space, spaces.MultiBinary):
            observation_space = spaces.MultiBinary(venv.observation_space.n * 2)
        else:
            raise ValueError("Error: observation space {} not supported for HER.".format(venv.observation_space))

        self.observation_space = observation_space
        super().__init__(venv, self.observation_space, venv.action_space)
        self.reward_function = reward_function
        self.actions = None
        self.goal = self.venv.observation_space.sample()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        obs, rew, done, info = self.venv.step(self.actions)
        rew = self.reward_function.get_reward(obs, self.actions, self.goal)
        # stack the goal to the observation and reshape it to the right size
        obs_goal = np.stack([obs, self.goal], axis=-1).reshape(obs.shape[:-1] + (obs.shape[-1] * 2,))
        return obs_goal, rew, done, info

    def reset(self):
        obs = self.venv.reset()
        self.goal = self.venv.observation_space.sample()
        # stack the goal to the observation and reshape it to the right size
        obs_goal = np.stack([obs, self.goal], axis=-1).reshape(obs.shape[:-1] + (obs.shape[-1] * 2,))
        return obs_goal

    def close(self):
        return

    def get_images(self):
        return self.venv.get_images()

    def render(self, *args, **kwargs):
        return self.venv.render(*args, **kwargs)
