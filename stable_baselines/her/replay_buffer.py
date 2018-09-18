import numpy as np

from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.her.utils import unstack_goal, stack_obs_goal


class HERBuffer(ReplayBuffer):
    """
    Base class for the Replay Buffer for HER.

    :param size: (int) The size of the buffer
    :param reward_func: (HERRewardFunctions) the reward function
    :param num_sample_goals: (int) the ratio of sampled HER to normal experience replay samples
    """

    def __init__(self, size, reward_func, num_sample_goals):
        super(HERBuffer, self).__init__(size=size)
        self.reward_func = reward_func
        self.num_sample_goals = num_sample_goals
        self.eps_count = 0  # the episode counter
        self.eps_idx = 0  # the frame in the episode
        self.eps_goal = []  # the goals in the current episode
        self.eps_goals = {0: self.eps_goal}  # the lookup table for the goals

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, eps_ids, eps_idx = [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, eps_id, eps_pos = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            eps_ids.append(eps_id)
            eps_idx.append(eps_pos)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), \
            np.array(eps_ids), np.array(eps_idx)

    def sample(self, batch_size, **kwargs):
        raise NotImplementedError()

    def add(self, obs_t, action, reward, obs_tp1, done):
        if done:
            # clean up unused goals
            self.eps_goals = {k: v for k, v in self.eps_goals.items()
                              if len(self._storage) <= self._next_idx or k >= self._storage[self._next_idx][5]}
            self.eps_count += 1
            self.eps_idx = 0
            self.eps_goal = []
            self.eps_goals[self.eps_count] = self.eps_goal
        data = (obs_t, action, reward, obs_tp1, done, self.eps_count, self.eps_idx)
        self.eps_idx += 1
        self.eps_goal.append(self._next_idx)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize


class FutureHERBuffer(HERBuffer):
    """
    HER Replay buffer that implements the Future strategy

    :param size: (int) The size of the buffer
    :param reward_func: (HERRewardFunctions) the reward function
    :param num_sample_goals: (int) the ratio of sampled HER to normal experience replay samples
    """
    def sample(self, batch_size, **kwargs):
        idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        future_proba = 1 - (1. / (1 + self.num_sample_goals))
        future_idx = np.random.uniform(size=batch_size) < future_proba
        obs_t, actions, rewards, obs_tp1, dones, eps_ids, eps_idx = self._encode_sample(idxes)

        for idx in np.where(future_idx)[0]:
            future_goal_idx = np.array(self.eps_goals[eps_ids[idx]])[eps_idx[idx]:]
            if future_goal_idx.shape[0] == 0:
                continue
            goal_idx = future_goal_idx[np.random.randint(future_goal_idx.shape[0])]
            goal_obs, _, _, _, _, _, _ = self._storage[goal_idx]
            goal_obs = unstack_goal(goal_obs)
            obs_t[idx] = stack_obs_goal(unstack_goal(obs_t[idx]), goal_obs)
            obs_tp1[idx] = stack_obs_goal(unstack_goal(obs_tp1[idx]), goal_obs)
            rewards[idx] = self.reward_func.get_reward(unstack_goal(obs_t[idx]), actions[idx], goal_obs)

        return obs_t, actions, rewards, obs_tp1, dones


class EpisodeHERBuffer(HERBuffer):
    """
    HER Replay buffer that implements the Episode strategy

    :param size: (int) The size of the buffer
    :param reward_func: (HERRewardFunctions) the reward function
    :param num_sample_goals: (int) the ratio of sampled HER to normal experience replay samples
    """
    def sample(self, batch_size, **kwargs):
        idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        episode_proba = 1 - (1. / (1 + self.num_sample_goals))
        episode_idx = np.random.uniform(size=batch_size) < episode_proba
        obs_t, actions, rewards, obs_tp1, dones, eps_ids, eps_idx = self._encode_sample(idxes)

        for idx in np.where(episode_idx)[0]:
            episode_goal_idx = np.array(self.eps_goals[eps_ids[idx]])
            if episode_goal_idx.shape[0] == 0:
                continue
            goal_idx = episode_goal_idx[np.random.randint(episode_goal_idx.shape[0])]
            goal_obs, _, _, _, _, _, _ = self._storage[goal_idx]
            goal_obs = unstack_goal(goal_obs)
            obs_t[idx] = stack_obs_goal(unstack_goal(obs_t[idx]), goal_obs)
            obs_tp1[idx] = stack_obs_goal(unstack_goal(obs_tp1[idx]), goal_obs)
            rewards[idx] = self.reward_func.get_reward(unstack_goal(obs_t[idx]), actions[idx], goal_obs)

        return obs_t, actions, rewards, obs_tp1, dones


class RandomHERBuffer(HERBuffer):
    """
    HER Replay buffer that implements the random strategy

    :param size: (int) The size of the buffer
    :param reward_func: (HERRewardFunctions) the reward function
    :param num_sample_goals: (int) the ratio of sampled HER to normal experience replay samples
    """
    def sample(self, batch_size, **kwargs):
        idxes = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        random_proba = 1 - (1. / (1 + self.num_sample_goals))
        random_idx = np.random.uniform(size=batch_size) < random_proba
        obs_t, actions, rewards, obs_tp1, dones, eps_ids, eps_idx = self._encode_sample(idxes)

        for idx in np.where(random_idx)[0]:
            if len(self._storage) == 0:
                break
            goal_idx = np.random.randint(len(self._storage))
            goal_obs, _, _, _, _, _, _ = self._storage[goal_idx]
            goal_obs = unstack_goal(goal_obs)
            obs_t[idx] = stack_obs_goal(unstack_goal(obs_t[idx]), goal_obs)
            obs_tp1[idx] = stack_obs_goal(unstack_goal(obs_tp1[idx]), goal_obs)
            rewards[idx] = self.reward_func.get_reward(unstack_goal(obs_t[idx]), actions[idx], goal_obs)

        return obs_t, actions, rewards, obs_tp1, dones


def make_her_buffer(buffer_class, reward_func, num_sample_goals):
    """
    Creates the Hindsight Experience Replay Buffer for HER

    :param buffer_class: (HERBuffer) the buffer type you wish to use
    :param reward_func: (HERRewardFunctions) The reward function to apply to the buffer
    :param num_sample_goals: (int) the number of goals to sample for every step
    """
    assert issubclass(buffer_class, HERBuffer), "Error: the buffer type, must be of type HERBuffer."

    class _Buffer(buffer_class):
        def __init__(self, size):
            super(_Buffer, self).__init__(size, reward_func, num_sample_goals)

    return _Buffer
