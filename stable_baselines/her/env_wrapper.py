from gym import spaces
import numpy as np

from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines.her.utils import stack_obs_goal


class HERWrapper(VecEnvWrapper):
    """
    Creates the wrapper for HER

    :param venv: (Gym environment) The environment to wrap
    :param reward_function: (HERRewardFunctions) The reward function to apply to the environment
    """
    def __init__(self, venv, reward_function):
        if isinstance(venv.observation_space, spaces.Discrete):
            observation_space = spaces.MultiDiscrete([venv.observation_space.n] * 2)
        elif isinstance(venv.observation_space, spaces.Box):
            shape = venv.observation_space.shape
            shape = np.array(shape[:-1] + (shape[-1] * 2,))

            low = venv.observation_space.low
            if not np.isscalar(low):
                low = np.array(list(low) * 2)
                shape = None # a quirk of Box, if low and high are not scalars, then shape must be None

            high = venv.observation_space.high
            if not np.isscalar(high):
                high = np.array(list(high) * 2)
                shape = None

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
        # stack the goal to the observation and reshape it to the right size
        obs_goal = stack_obs_goal(obs, self.goal)
        return obs_goal, rew, done, info

    def reset(self):
        obs = self.venv.reset()
        self.goal = self.venv.observation_space.sample()
        # stack the goal to the observation and reshape it to the right size
        obs_goal = stack_obs_goal(obs, self.goal)
        return obs_goal

    def close(self):
        return

    def get_images(self):
        return self.venv.get_images()

    def render(self, *args, **kwargs):
        return self.venv.render(*args, **kwargs)
