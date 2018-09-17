from abc import ABC, abstractmethod

from gym import spaces
import numpy as np


class HERRewardFunctions(ABC):
    """
    The base class for HER reward functions
    """
    def __init__(self):
        self.env = None

    @abstractmethod
    def set_env(self, env):
        """
        sets the environment for the reward func

        :param env: (Gym environment) the environment that uses this reward function
            (should you need more information from the environment)
        """
        pass

    @abstractmethod
    def get_reward(self, observation, action, goal):
        """
        Returns the reward for a given observation, action and target goal.

        :param observation: (np.ndarray) the observation of the environment
        :param action: (np.ndarray) the taken action for the environment
        :param goal: (np.ndarray) the target goal
        :return: (float) the reward for the given transition
        """
        pass


class ProximalReward(HERRewardFunctions):
    """
    Return a positive reward if the target was reached, within a specific interval

    :param eps: (float) the epsilon value for where proximity is considered to be true
    """
    def __init__(self, eps):
        super(ProximalReward, self).__init__()
        self.eps = eps
        self.float_comp = None

    def set_env(self, env):
        self.env = env
        if isinstance(self.env.observation_space, spaces.Box):
            self.float_comp = True
        elif (isinstance(self.env.observation_space, spaces.MultiDiscrete) or
              isinstance(self.env.observation_space, spaces.Discrete) or
              isinstance(self.env.observation_space, spaces.MultiBinary)):
            self.float_comp = False
        else:
            raise ValueError("Error: observation space {} not supported for the ProximalReward function."
                             .format(self.env.observation_space))

    def get_reward(self, observation, action, goal):
        if self.env is None:
            raise ValueError("Undefined environment! halting.")
        if self.float_comp:
            return 1 if np.all(np.abs(observation - goal) <= self.eps) else -1
        else:
            return 1 if np.all(observation == goal) else -1
