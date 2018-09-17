from stable_baselines.common import OffPolicyRLModel
from stable_baselines.her.replay_buffer import make_her_buffer
from stable_baselines.her.env_wrapper import HERWrapper


class HER(OffPolicyRLModel):
    """
    The HER (Hindsight Experience Replay) model class, https://arxiv.org/abs/1707.01495

    :param model: (OffPolicyRLModel) The off policy RL model to apply Hindsight Experience Replay
    :param policy: (BasePolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param reward_function: (HERRewardFunctions) the reward function for HER
    :param num_sample_goals: (int) the number of goals to sample for every step
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param *args: positional arguments for the model
    :param **kwargs: keyword arguments for the model
    """
    def __init__(self, model, policy, env, reward_function, num_sample_goals=4, verbose=0, _init_setup_model=True,
                 *args, **kwargs):
        super(HER, self).__init__(policy=None, env=env, replay_buffer=None,
                                  verbose=verbose, policy_base=None, requires_vec_env=True)

        assert issubclass(model, OffPolicyRLModel), \
            "Error: HER only works with Off policy model (such as DDPG and DQN)."

        self.reward_function = reward_function
        self.model_class = model

        if self.env is not None:
            env = HERWrapper(self.env, reward_function)
        replay_buffer = make_her_buffer(reward_function, num_sample_goals)
        self.model = model(policy=policy, env=env, verbose=verbose, replay_buffer=replay_buffer,
                           _init_setup_model=_init_setup_model, *args, **kwargs)

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.reward_function.set_env(self.env, self.observation_space)
        self.model.setup_model()

    def set_env(self, env):
        super().set_env(env)
        self.reward_function.set_env(self.env, self.observation_space)
        if self.env is not None:
            env = HERWrapper(self.env, self.reward_function)
        self.model.set_env(env)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name=None):
        if tb_log_name is None:
            tb_log_name = "HER_{}".format(self.model_class.__name__)
        self.model.learn(total_timesteps=total_timesteps, callback=callback, seed=seed, log_interval=log_interval,
                         tb_log_name=tb_log_name)
        return self

    def predict(self, observation, state=None, mask=None, deterministic=False):
        return self.model.predict(observation, state=state, mask=mask, deterministic=deterministic)

    def action_probability(self, observation, state=None, mask=None):
        return self.model.action_probability(observation, state=state, mask=mask)

    def get_save_data(self):
        return {
            "model_class": self.model_class,
            "reward_function": self.reward_function,
            "observation_space": self.observation_space,
        }

    def save(self, save_path):
        data = self.get_save_data()
        model_data = self.model.get_save_data()

        params = self.model.sess.run(self.model.params)

        self._save_to_file(save_path, data=(data, model_data), params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        (data, model_data), params = cls._load_from_file(load_path)

        her_model = cls(model=data["model_class"], policy=model_data["policy"],
                        reward_function=data["reward_function"], env=None, _init_setup_model=False)
        her_model.__dict__.update(data)
        her_model.model.__dict__.update(model_data)
        her_model.model.__dict__.update(kwargs)
        her_model.set_env(env)
        her_model.setup_model()

        restores = []
        for param, loaded_p in zip(her_model.model.params, params):
            restores.append(param.assign(loaded_p))
        her_model.model.sess.run(restores)

        return her_model
