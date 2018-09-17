import numpy as np

from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.her.utils import unstack_goal, stack_obs_goal


def make_her_buffer(reward_func, num_sample_goals):
    """
    Creates the Hindsight Experience Replay Buffer for HER

    :param reward_func: (HERRewardFunctions) The reward function to apply to the buffer
    :param num_sample_goals: (int) the number of goals to sample for every step
    """
    class HindsightExperienceReplayBuffer(ReplayBuffer):
        """
        Replay Buffer for HER

        :param size: (int) The size of the buffer
        """
        def __init__(self, size):
            super(HindsightExperienceReplayBuffer, self).__init__(size=size)
            self.reward_func = reward_func
            self.num_sample_goals = num_sample_goals

        def add(self, obs_t, action, reward, obs_tp1, done):
            super().add(obs_t, action, reward, obs_tp1, done)

            start = None
            length = 0
            for i in range(1, self._maxsize + 1):
                # walk backwards to know the range of the episode
                step = (self._next_idx - i) % (self._maxsize - 1)

                if step == 0 and len(self._storage) < (self._maxsize - 1):
                    start = 0
                    length = self._next_idx
                    break
                elif self._storage[step][4]:  # if end of episode
                    start = step
                    length = (start + self._next_idx) % (self._maxsize - 1)

            if start is None:
                start = 0
                length = self._maxsize

            # sample goals randomly withing the current episode
            goals = [self._storage
                     [(np.random.randint(0, length) + start) % (self._maxsize - 1)]
                     [0]
                     [:obs_t.shape[-1] // 2]
                     for _ in range(num_sample_goals)]

            # stack the new goals to the current obs, obs+1 and recalculate the reward
            for goal in goals:
                obs_t_, obs_tp1_, reward_ = \
                    (stack_obs_goal(unstack_goal(obs_t), goal),
                     stack_obs_goal(unstack_goal(obs_tp1), goal),
                     reward_func.get_reward(unstack_goal(obs_t), action, goal))
                super().add(obs_t_, action, reward_, obs_tp1_, done)

    return HindsightExperienceReplayBuffer
