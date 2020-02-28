import numpy as np

from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer


def test_extend_uniform():
    nvals = 16
    states = [np.random.rand(2, 2) for _ in range(nvals)]
    actions = [np.random.rand(2) for _ in range(nvals)]
    rewards = [np.random.rand() for _ in range(nvals)]
    newstate = [np.random.rand(2, 2) for _ in range(nvals)]
    done = [np.random.randint(0, 2) for _ in range(nvals)]

    size = 32
    baseline = ReplayBuffer(size)
    ext = ReplayBuffer(size)
    for data in zip(states, actions, rewards, newstate, done):
        baseline.add(*data)

    states, actions, rewards, newstates, done = map(
        np.array, [states, actions, rewards, newstate, done])

    ext.extend(states, actions, rewards, newstates, done)
    assert len(baseline) == len(ext)

    # Check buffers have same values
    for i in range(nvals):
        for j in range(5):
            condition = (baseline.storage[i][j] == ext.storage[i][j])
            if isinstance(condition, np.ndarray):
                # for obs, obs_t1
                assert np.all(condition)
            else:
                # for done, reward action
                assert condition


def test_extend_prioritized():
    nvals = 16
    states = [np.random.rand(2, 2) for _ in range(nvals)]
    actions = [np.random.rand(2) for _ in range(nvals)]
    rewards = [np.random.rand() for _ in range(nvals)]
    newstate = [np.random.rand(2, 2) for _ in range(nvals)]
    done = [np.random.randint(0, 2) for _ in range(nvals)]

    size = 32
    alpha = 0.99
    baseline = PrioritizedReplayBuffer(size, alpha)
    ext = PrioritizedReplayBuffer(size, alpha)
    for data in zip(states, actions, rewards, newstate, done):
        baseline.add(*data)

    states, actions, rewards, newstates, done = map(
        np.array, [states, actions, rewards, newstate, done])

    ext.extend(states, actions, rewards, newstates, done)
    assert len(baseline) == len(ext)

    # Check buffers have same values
    for i in range(nvals):
        for j in range(5):
            condition = (baseline.storage[i][j] == ext.storage[i][j])
            if isinstance(condition, np.ndarray):
                # for obs, obs_t1
                assert np.all(condition)
            else:
                # for done, reward action
                assert condition

    # assert priorities
    assert (baseline._it_min._value == ext._it_min._value).all()
    assert (baseline._it_sum._value == ext._it_sum._value).all()
