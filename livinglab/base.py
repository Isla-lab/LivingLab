import numpy as np

from abc import ABC
from typing import Any, Optional, Mapping


class Environment(ABC):
    def __init__(self, seed: Optional[int], start_time_step: int, end_time_step: int, episode_length: Optional[int]=None):
        self.seed = seed
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step
        self.episode_length = episode_length if episode_length is not None else end_time_step - start_time_step

    @property
    def seed(self):
        return self._seed
    
    @property
    def time_step(self):
        return self._time_step
    
    @property
    def start_time_step(self):
        return self._start_time_step
    
    @property
    def end_time_step(self):
        return self._end_time_step
    
    @property
    def episode_length(self):
        return self._episode_length
    
    @seed.setter
    def seed(self, new_seed: int):
        self._seed = np.random.randint(low=0, high=1_000_000) if new_seed is None else new_seed

    @time_step.setter
    def time_step(self, new_step: int):
        assert self._start_time_step <= new_step <= self._end_time_step, \
            f'Invalid current time step (current={new_step} not in [{self._start_time_step}, {self._end_time_step}]).'
        self._time_step = new_step

    @start_time_step.setter
    def start_time_step(self, new_step: int):
        self._start_time_step = new_step

    @end_time_step.setter
    def end_time_step(self, new_step: int):
        assert new_step > self._start_time_step, \
            f'Invalid simulation start/end steps (start={self._start_time_step} >= end={new_step}).'
        self._end_time_step = new_step

    @episode_length.setter
    def episode_length(self, new_len: int):
        if new_len is None:
            self._episode_length = (self._end_time_step - self._start_time_step) + 1
        else:
            self._episode_length = new_len

    def step(self):
        self.time_step += 1

    def reset(self):
        self.time_step = 0


class Device(Environment):
    """
    Device base class.

    Parameters
    ----------
    :param efficiency: Technical efficiency.
    :type efficiency: float
    """
    def __init__(self, efficiency: float, **kwargs: Mapping[str, Any]):
        super().__init__(**kwargs)
        self.efficiency = efficiency

    @property
    def efficiency(self):
        return self._efficiency
    
    @efficiency.setter
    def efficiency(self, new_eff: float):
        assert new_eff is None or new_eff >= 0, f'Invalid efficiency {new_eff}. Must be >= 0.'
        self._efficiency = new_eff