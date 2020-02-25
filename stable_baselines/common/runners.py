from abc import ABC, abstractmethod
import typing
from typing import Union, Optional, Any

import gym
import numpy as np

from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import VecEnv

if typing.TYPE_CHECKING:
    from stable_baselines.common.base_class import BaseRLModel  # pytype: disable=pyi-error


class AbstractEnvRunner(ABC):
    def __init__(self, *, env: Union[gym.Env, VecEnv], model: 'BaseRLModel', n_steps: int):
        """
        Collect experience by running `n_steps` in the environment.
        Note: if this is a `VecEnv`, the total number of steps will
        be `n_steps * n_envs`.

        :param env: (Union[gym.Env, VecEnv]) The environment to learn from
        :param model: (BaseRLModel) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        """
        self.env = env
        self.model = model
        n_envs = env.num_envs
        self.batch_ob_shape = (n_envs * n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_envs,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_envs)]
        self.callback = None  # type: Optional[BaseCallback]
        self.continue_training = True
        self.n_envs = n_envs

    def run(self, callback: Optional[BaseCallback] = None) -> Any:
        """
        Collect experience.

        :param callback: (Optional[BaseCallback]) The callback that will be called
            at each environment step.
        """
        self.callback = callback
        self.continue_training = True
        return self._run()

    @abstractmethod
    def _run(self) -> Any:
        """
        This method must be overwritten by child class.
        """
        raise NotImplementedError()
