"""This file is used for specifying various schedules that evolve over
time throughout the execution of the algorithm, such as:

 - learning rate for the optimizer
 - exploration epsilon for the epsilon greedy exploration strategy
 - beta parameter for beta parameter in prioritized replay

Each schedule has a function `value(t)` which returns the current value
of the parameter given the timestep t of the optimization procedure.
"""


class Schedule(object):
    def value(self, step):
        """
        Value of the schedule for a given timestep

        :param step: (int) the timestep
        :return: (float) the output value for the given timestep
        """
        raise NotImplementedError


class ConstantSchedule(Schedule):
    """
    Value remains constant over time.

    :param value: (float) Constant value of the schedule
    """

    def __init__(self, value):
        self._value = value

    def value(self, step):
        return self._value


def linear_interpolation(left, right, alpha):
    """
    Linear interpolation between `left` and `right`.

    :param left: (float) left boundary
    :param right: (float) right boundary
    :param alpha: (float) coeff in [0, 1]
    :return: (float)
    """

    return left + alpha * (right - left)


class PiecewiseSchedule(Schedule):
    """
    Piecewise schedule.

    :param endpoints: ([(int, int)])
        list of pairs `(time, value)` meaning that schedule should output
        `value` when `t==time`. All the values for time must be sorted in
        an increasing order. When t is between two times, e.g. `(time_a, value_a)`
        and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
        `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
        time passed between `time_a` and `time_b` for time `t`.
    :param interpolation: (lambda (float, float, float): float)
        a function that takes value to the left and to the right of t according
        to the `endpoints`. Alpha is the fraction of distance from left endpoint to
        right endpoint that t has covered. See linear_interpolation for example.
    :param outside_value: (float)
        if the value is requested outside of all the intervals specified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested.
    """

    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, step):
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= step < right_t:
                alpha = float(step - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(Schedule):
    """
    Linear interpolation between initial_p and final_p over
    schedule_timesteps. After this many timesteps pass final_p is
    returned.

    :param schedule_timesteps: (int) Number of timesteps for which to linearly anneal initial_p to final_p
    :param initial_p: (float) initial output value
    :param final_p: (float) final output value
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, step):
        fraction = min(float(step) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constfn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule


def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func


# ================================================================
# Legacy scheduler used by A2C, AKCTR and ACER
# ================================================================

def constant(_):
    """
    Returns a constant value for the Scheduler

    :param _: ignored
    :return: (float) 1
    """
    return 1.


def linear_schedule(progress):
    """
    Returns a linear value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress
    """
    return 1 - progress


def middle_drop(progress):
    """
    Returns a linear value with a drop near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress if (1 - progress) >= 0.75 else 0.075
    """
    eps = 0.75
    if 1 - progress < eps:
        return eps * 0.1
    return 1 - progress


def double_linear_con(progress):
    """
    Returns a linear value (x2) with a flattened tail for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) 1 - progress*2 if (1 - progress*2) >= 0.125 else 0.125
    """
    progress *= 2
    eps = 0.125
    if 1 - progress < eps:
        return eps
    return 1 - progress


def double_middle_drop(progress):
    """
    Returns a linear value with two drops near the middle to a constant value for the Scheduler

    :param progress: (float) Current progress status (in [0, 1])
    :return: (float) if 0.75 <= 1 - p: 1 - p, if 0.25 <= 1 - p < 0.75: 0.75, if 1 - p < 0.25: 0.125
    """
    eps1 = 0.75
    eps2 = 0.25
    if 1 - progress < eps1:
        if 1 - progress < eps2:
            return eps2 * 0.5
        return eps1 * 0.1
    return 1 - progress


SCHEDULES = {
    'linear': linear_schedule,
    'constant': constant,
    'double_linear_con': double_linear_con,
    'middle_drop': middle_drop,
    'double_middle_drop': double_middle_drop
}


class Scheduler(object):
    def __init__(self, initial_value, n_values, schedule):
        """
        Update a value every iteration, with a specific curve.

        This is a legacy version of schedules, originally defined
        in a2c/utils.py. Used by A2C, ACER and ACKTR algorithms.

        :param initial_value: (float) initial value
        :param n_values: (int) the total number of iterations
        :param schedule: (function) the curve you wish to follow for your value
        """
        self.step = 0.
        self.initial_value = initial_value
        self.nvalues = n_values
        self.schedule = SCHEDULES[schedule]

    def value(self):
        """
        Update the Scheduler, and return the current value

        :return: (float) the current value
        """
        current_value = self.initial_value * self.schedule(self.step / self.nvalues)
        self.step += 1.
        return current_value

    def value_steps(self, steps):
        """
        Get a value for a given step

        :param steps: (int) The current number of iterations
        :return: (float) the value for the current number of iterations
        """
        return self.initial_value * self.schedule(steps / self.nvalues)