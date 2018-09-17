import numpy as np


def stack_obs_goal(obs, goal):
    """
    stack the observation to the goal

    :param obs: (np.ndarray) the observation
    :param goal: (np.ndarray) the goal
    :return: (np.ndarray) the stacked observation - goal
    """
    return np.stack([obs, goal.reshape(obs.shape)], axis=-1).reshape(obs.shape[:-1] + (obs.shape[-1] * 2,))


def unstack_goal(stacked_obs):
    """
    unstacks the observation from the goal

    :param stacked_obs: (np.ndarray) the stacked observation - goal
    :return: (np.ndarray) the observation
    """
    return stacked_obs[:stacked_obs.shape[-1] // 2]
