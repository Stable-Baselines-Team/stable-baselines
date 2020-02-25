import os
import shutil

import pytest

from stable_baselines import A2C, ACKTR, ACER, DQN, DDPG, PPO1, PPO2, SAC, TD3, TRPO
from stable_baselines.common.callbacks import (CallbackList, CheckpointCallback, EvalCallback,
    EveryNTimesteps, StopTrainingOnRewardThreshold, BaseCallback)


LOG_FOLDER = './logs/callbacks/'


class CustomCallback(BaseCallback):
    """
    Callback to check that every method was called once at least
    """
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.calls = {
            'training_start': False,
            'rollout_start': False,
            'step': False,
            'rollout_end': False,
            'training_end': False,
        }

    def _on_training_start(self):
        self.calls['training_start'] = True

    def _on_rollout_start(self):
        self.calls['rollout_start'] = True

    def _on_step(self):
        self.calls['step'] = True
        return True

    def _on_rollout_end(self):
        self.calls['rollout_end'] = True

    def _on_training_end(self):
        self.calls['training_end'] = True

    def validate(self, allowed_failures):
        for allowed_failure in allowed_failures:
            self.calls[allowed_failure] = True
        assert all(self.calls.values())


@pytest.mark.parametrize("model_class", [A2C, ACER, ACKTR, DQN, DDPG, PPO1, PPO2, SAC, TD3, TRPO])
def test_callbacks(model_class):

    env_id = 'Pendulum-v0'
    if model_class in [ACER, DQN]:
        env_id = 'CartPole-v1'

    allowed_failures = []
    # Number of training timesteps is too short
    # otherwise, the training would take too long, or would require
    # custom parameter per algorithm
    if model_class in [PPO1, DQN, TRPO]:
        allowed_failures = ['rollout_end']

    # Create RL model
    model = model_class('MlpPolicy', env_id)

    checkpoint_callback = CheckpointCallback(save_freq=500, save_path=LOG_FOLDER)

    # For testing: use the same training env
    eval_env = model.get_env()
    # Stop training if the performance is good enough
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-1200, verbose=1)

    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best,
                                 best_model_save_path=LOG_FOLDER,
                                 log_path=LOG_FOLDER, eval_freq=100)

    # Equivalent to the `checkpoint_callback`
    # but here in an event-driven manner
    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path=LOG_FOLDER,
                                             name_prefix='event')
    event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

    callback = CallbackList([checkpoint_callback, eval_callback, event_callback])

    model.learn(500, callback=callback)
    model.learn(200, callback=None)
    custom_callback = CustomCallback()
    model.learn(200, callback=custom_callback)
    # Check that every called were executed
    custom_callback.validate(allowed_failures=allowed_failures)
    # Transform callback into a callback list automatically
    custom_callback = CustomCallback()
    model.learn(500, callback=[checkpoint_callback, eval_callback, custom_callback])
    # Check that every called were executed
    custom_callback.validate(allowed_failures=allowed_failures)

    # Automatic wrapping, old way of doing callbacks
    model.learn(200, callback=lambda _locals, _globals : True)

    # Cleanup
    if os.path.exists(LOG_FOLDER):
        shutil.rmtree(LOG_FOLDER)
