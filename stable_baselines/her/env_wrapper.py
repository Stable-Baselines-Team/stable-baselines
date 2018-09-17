from gym import spaces

from stable_baselines.common.vec_env.base_vec_env import VecEnvWrapper


class HERWrapper(VecEnvWrapper):
    def __init__(self, venv):
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

        observation_space = observation_space
        super().__init__(venv, observation_space, venv.action_space)

    def reset(self):
        pass

    def step_wait(self):
        pass
