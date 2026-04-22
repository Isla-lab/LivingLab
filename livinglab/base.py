import numpy as np

from abc import ABC
from typing import Any, Optional, Mapping


class Environment(ABC):
    """
    Environment element base class.

    Parameters
    ----------
    :param seed: Experiment seed for reproducibility.
    :type seed: Optional[int]
    :start_time_step: Simulation start time step.
    :type start_time_step: Optional[int]
    :end_time_step: Simulation end time step.
    :type end_time_step: Optional[int]
    :param episode_length: Time steps duration of a simulation episode.
    :type episode_length: Optional[int]
    """
    def __init__(self, seed: Optional[int]=None, start_time_step: Optional[int]=None, end_time_step: Optional[int]=None, episode_length: Optional[int]=None):
        self.seed = seed

        # Checks
        assert not (start_time_step is None and end_time_step is not None), f'Invalid time steps initialization. Initialize both time steps or set both `None`.'
        assert not (start_time_step is not None and end_time_step is None), f'Invalid time steps initialization. Initialize both time steps or set both `None`.'

        # Simulation info
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step

        # Episodic info
        self.episode_length = episode_length

    @property
    def seed(self) -> int:
        """Experiment seed for reproducibility."""
        return self._seed
    
    @property
    def time_step(self) -> int:
        """Current absolute time step."""
        return self._episode_start_time_step + self._episode_time_step
    
    @property
    def start_time_step(self) -> int:
        """Simulation start time step."""
        return self._start_time_step
    
    @property
    def end_time_step(self) -> int:
        """Simulation end time step."""
        return self._end_time_step
    
    @property
    def episode_length(self):
        """Time steps duration of a simulation episode."""
        return self._episode_length
    
    @property
    def simulation_episodes(self):
        """Number of simulation episodes."""
        return self._simulation_episodes

    @property
    def episode_counter(self):
        """Episode counter for sound `self.episode_start_time_step` and `self.episode_end_time_step` retrieval."""
        return self._episode_counter
    
    @property
    def episode_time_step(self):
        """Current simulation episode time step."""
        return self._episode_time_step

    @property
    def episode_start_time_step(self):
        """Simulation episode start time step."""
        return self._episode_start_time_step
    
    @property
    def episode_end_time_step(self):
        """Simulation episode end time step."""
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
    def start_time_step(self, new_step: Optional[int]):
        if hasattr(self, '_end_time_step'):
            if self._end_time_step is not None:
                assert new_step is None or new_step < self._end_time_step, \
                    f'Invalid simulation start/end steps (start={new_step} >= end={self._end_time_step}).'
            
        self._start_time_step = new_step

    @end_time_step.setter
    def end_time_step(self, new_step: Optional[int]):
        if hasattr(self, '_start_time_step'):
            if self._start_time_step is not None:
                assert new_step is None or new_step > self._start_time_step, \
                    f'Invalid simulation start/end steps (start={self._start_time_step} >= end={new_step}).'
        
        self._end_time_step = new_step

    @episode_length.setter
    def episode_length(self, new_len: Optional[int]):
        if self._start_time_step is not None and self._end_time_step is not None:
            if new_len is None:
                self._episode_length = (self._end_time_step - self._start_time_step) + 1
            else:
                self._episode_length = new_len

            if (self._end_time_step + 1) % self._episode_length != 0:
                print(f'[WARN] Episode length {self._episode_length} does not assure a full simulation coverage.')
        else:
            if new_len is not None:
                print('[WARN] Trying to set new episode length with no `start_time_step` and `end_time_step`. Automatically set to None.')
            self._episode_length = None

        if self._episode_length is not None:    
            self.simulation_episodes = (self._end_time_step + 1) // self.episode_length
            self.episode_counter = -1

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
        """
        Reset the environment to its initial state, 
        and set next `self.episode_start_time_step` and `self.episode_end_time_step` correclty.
        """
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
    :param **kwargs: Keyword arguments to initialize `Environment`.
    :type **kwargs: Mapping[str, Any]
    """
    def __init__(self, efficiency: float, **kwargs: Mapping[str, Any]):
        super().__init__(**kwargs)
        self.efficiency = efficiency

    @property
    def efficiency(self) -> float:
        """Device's technical efficiency."""
        return self._efficiency
    
    @efficiency.setter
    def efficiency(self, new_eff: float):
        assert new_eff is None or new_eff >= 0, f'Invalid efficiency {new_eff}. Must be >= 0.'
        self._efficiency = new_eff