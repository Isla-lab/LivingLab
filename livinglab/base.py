import numpy as np

from abc import ABC
from typing import Any, Optional, Mapping


class Environment(ABC):
    def __init__(self, seed: Optional[int], start_time_step: int, end_time_step: int, episode_length: Optional[int]=None):
        self.seed = seed

        # Simulation info
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step

        # Episodic info
        self.episode_length = episode_length        
        self.simulation_episodes = (self.end_time_step + 1) // self.episode_length
        self.episode_counter = -1

    @property
    def seed(self):
        return self._seed
    
    @property
    def time_step(self):
        return self._episode_start_time_step + self._episode_time_step
    
    @property
    def start_time_step(self):
        return self._start_time_step
    
    @property
    def end_time_step(self):
        return self._end_time_step
    
    @property
    def episode_length(self):
        return self._episode_length
    
    @property
    def simulation_episodes(self):
        return self._simulation_episodes

    @property
    def episode_counter(self):
        return self._episode_counter
    
    @property
    def episode_time_step(self):
        return self._episode_time_step

    @property
    def episode_start_time_step(self):
        return self._episode_start_time_step
    
    @property
    def episode_end_time_step(self):
        return self._episode_end_time_step
    
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

        if (self._end_time_step + 1) % self._episode_length != 0:
            print(f'[WARN] Episode length {self._episode_length} does not assure a full simulation coverage.')

    @simulation_episodes.setter
    def simulation_episodes(self, n: int):
        assert n > 0, f'Invalid number of simulation episodes n={n}. Must be > 0.'
        self._simulation_episodes = n

    @episode_counter.setter
    def episode_counter(self, new_value: int):
        self._episode_counter = new_value

    @episode_time_step.setter
    def episode_time_step(self, new_step: int):
        assert new_step < self._episode_length, f'Episode running time step exceeding episode length (timestep={new_step} >= length={self._episode_length}).'
        self._episode_time_step = new_step

    @episode_start_time_step.setter
    def episode_start_time_step(self, new_step: int):
        assert self._start_time_step <= new_step <= self._end_time_step, \
            f'Invalid episode start time step: {new_step} not in [{self._start_time_step}, {self._end_time_step}]'
        self._episode_start_time_step = new_step
    
    @episode_end_time_step.setter
    def episode_end_time_step(self, new_step: int):
        assert self._start_time_step <= new_step <= self._end_time_step, \
            f'Invalid episode end time step: {new_step} not in [{self._start_time_step}, {self._end_time_step}]'
        assert new_step > self._episode_start_time_step, \
            f'Invalid episode start/end steps (start={self._episode_start_time_step} >= end={new_step}).'
        self._episode_end_time_step = new_step

    def step(self):
        self.episode_time_step += 1

    def reset(self):
        self.episode_counter += 1
        self.episode_time_step = 0
        self.episode_start_time_step = (self.episode_counter % self.simulation_episodes) * self.episode_length
        self.episode_end_time_step = self.episode_start_time_step + self.episode_length - 1


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