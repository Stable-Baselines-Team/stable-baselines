import pytest
import numpy as np

from stable_baselines import A2C, ACKTR, PPO1, PPO2, TRPO
from stable_baselines.common.identity_env import IdentityEnvBox


class Helper:
    @staticmethod
    def proba_vals(obs, state, mask):
        # Return fixed mean, std
        return np.array([-0.4]), np.array([[0.1]])


@pytest.mark.parametrize("model_class", [A2C, ACKTR, PPO1, PPO2, TRPO])
def test_log_prob_calcuation(model_class):
    model = model_class("MlpPolicy", IdentityEnvBox())
    # Fixed mean/std
    model.proba_step = Helper.proba_vals
    # Check that the log probability is the one expected for the given mean/std
    logprob = model.action_probability(observation=np.array([[0.5], [0.5]]), actions=0.2, logp=True)
    assert np.allclose(logprob, np.array([-16.616353440210627])), "Calculation failed for {}".format(model_class)
