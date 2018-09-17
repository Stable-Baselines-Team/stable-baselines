from stable_baselines.common import OffPolicyRLModel
from stable_baselines.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines.her.replay_buffer import make_her_buffer
from stable_baselines.her.env_wrapper import HERWrapper


class HER(OffPolicyRLModel):
    """

    :param model: (OffPolicyRLModel)
    :param policy: (BasePolicy)
    :param env: (Gym Environment)
    :param reward_function: (HERRewardFunctions)
    :param verbose: (int)
    :param _init_setup_model: (bool)
    :param *args: positional arguments for the model
    :param **kwargs: keyword arguments for the model
    """
    def __init__(self, model, policy, env, reward_function, verbose=0, _init_setup_model=False, *args, **kwargs):
        super(HER, self).__init__(policy=None, env=env, replay_buffer=None,
                                  verbose=verbose, policy_base=None, requires_vec_env=False)

        assert issubclass(model, OffPolicyRLModel), \
            "Error: HER only works with Off policy model (such as DDPG and DQN)."

        self.reward_function = reward_function
        self.model_class = model

        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        env = HERWrapper(env, reward_function)
        self.model = model(policy=policy, env=env, verbose=verbose, replay_buffer=make_her_buffer(reward_function),
                           *args, **kwargs)

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.reward_function.set_env(self.env)
        self.model.setup_model()

    def set_env(self, env):
        super().set_env(env)
        self.reward_function.set_env(env)
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
        }

    def save(self, save_path):
        data = self.get_save_data()
        model_data = self.model.get_save_data()

        params = self.model.sess.run(self.model.params)

        self._save_to_file(save_path, data=(data, model_data), params=params)

    @classmethod
    def load(cls, load_path, env=None, **kwargs):
        (data, model_data), params = cls._load_from_file(load_path)

        her_model = cls(model=data["model_class"], policy=model_data["policy"], env=None, _init_setup_model=False)
        her_model.model.__dict__.update(model_data)
        her_model.model.__dict__.update(kwargs)
        her_model.model.set_env(env)
        her_model.setup_model()

        restores = []
        for param, loaded_p in zip(her_model.model.params, params):
            restores.append(param.assign(loaded_p))
        her_model.model.sess.run(restores)

        return her_model
