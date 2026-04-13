from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Mapping


class RewardFunction(ABC):
    """
    Base reward function class.

    Parameters
    ----------
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any] 
    """
    def __init__(self, env_metadata: Optional[Mapping[str, Any]]=None):
        self.env_metadata = env_metadata

    @property
    def env_metadata(self) -> Mapping[str, Any]:
        """Environment static information."""
        return self._env_metadata
    
    @env_metadata.setter
    def env_metadata(self, new_metadata: Mapping[str, Any]):
        self._env_metadata = new_metadata

    @abstractmethod
    def calculate(self, observations: Mapping[str, Union[int, float]]) -> float:
        """
        Compute the reward given the current `time_step` observations after applying actions to the environment.

        Parameters
        ----------
        :param observations: Current named observations.
        :type observations: Mapping[str, Union[int, float]]
        """
        pass


class ComfortRewardFuction(RewardFunction):
    """
    Thermal comfort reward function.

    The reward is defined as the absoulte difference between the measured indoor dry-bulb temperature and the setpoint.

    This difference is raised to some `exponent` if exceeding a given `comfort_band`.

    Parameters
    ----------
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any]
    :param comfort_band: Setpoint comfort difference (+/-).
    :type comfort_band: Optional[float]
    :param exponent: Exponent to raise the temperature difference to if exceeding `comfort_band`.
    :type exponent: Optional[float] 
    """
    def __init__(self, env_metadata: Optional[Mapping[str, Any]]=None, comfort_band: Optional[float]=None, exponent: Optional[float]=None):
        super().__init__(env_metadata)

        self.comfort_band = comfort_band
        self.exponent = exponent

    @property
    def exponent(self) -> float:
        """Exponent to raise the temperature difference to if exceeding `self.comfort_band`."""
        return self._exponent
    
    @property
    def comfort_band(self) -> float:
        """Setpoint comfort difference (+/-)."""
        return self._comfort_band
    
    @exponent.setter
    def exponent(self, new_exp: Optional[float]):
        self._exponent = 2.0 if new_exp is None else new_exp

    @comfort_band.setter
    def comfort_band(self, new_band: Optional[float]):
        self._comfort_band = new_band

    def calculate(self, observations: Mapping[str, Union[int, float]]) -> float:
        # Retrieve required observations
        indoor_dry_bulb_temperature = observations['indoor_dry_bulb_temperature']
        indoor_dry_bulb_temperature_cooling_set_point = observations['indoor_dry_bulb_temperature_cooling_set_point']
        comfort_band = observations.get('comfort_band', None) if self.comfort_band is None else self.comfort_band
        assert comfort_band is not None, f'Comfort band is required for {self.__class__.__name__}.calculate(). None has been provided.'

        # Compute reward
        temp_delta = abs(indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_cooling_set_point)
        reward = -(temp_delta**self.exponent) if temp_delta >= comfort_band else -temp_delta

        return reward