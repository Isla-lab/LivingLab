import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Tuple, List, Mapping


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


class ComfortRewardFunction(RewardFunction):
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
        hp_mode = self.env_metadata['heat_pump']['mode']
        indoor_dry_bulb_temperature = observations['indoor_dry_bulb_temperature']
        indoor_dry_bulb_temperature_set_point = observations[f'indoor_dry_bulb_temperature_{hp_mode}_set_point']
        comfort_band = observations.get('comfort_band', None) if self.comfort_band is None else self.comfort_band
        assert comfort_band is not None, f'Comfort band is required for {self.__class__.__name__}.calculate(). None has been provided.'

        # Compute temperature delta
        if hp_mode == 'cooling':
            temp_delta = indoor_dry_bulb_temperature - indoor_dry_bulb_temperature_set_point
        else:
            temp_delta = indoor_dry_bulb_temperature_set_point - indoor_dry_bulb_temperature

        # Compute reward
        if -comfort_band <= temp_delta <= 0.0:
            reward = 0.0
        else:
            temp_delta = abs(temp_delta)
            reward = -(temp_delta**self.exponent) if temp_delta > comfort_band else -temp_delta

        return reward
    
    
class NetElectricityConsumptionFunction(RewardFunction):
    """
    Net electricity consumption reward.

    This reward is designed to penalize large values of electricity imported from the grid.
        
    Parameters
    ----------
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any]
    """
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: Mapping[str, Union[int, float]]) -> float:

        # Electricity consumption
        e = observations['net_electricity_consumption']
            
        reward = -max(0.0, e)
        return reward
    

class ElectricityCostRewardFunction(RewardFunction):
    """
    Electicity import monetary cost reward.

    This reward is designed to penalize high costs due to electricity consumption.
        
    Parameters
    ----------
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any]
    """
    def __init__(self, env_metadata: Mapping[str, Any]):
        super().__init__(env_metadata)

    def calculate(self, observations: Mapping[str, Union[int, float]]) -> float:

        # Electricity consumption
        e = observations['net_electricity_consumption']
        p = observations['electricity_pricing']
            
        reward = -max(0.0, e*p*100.0)
        return reward
        
        
class NetElectricityConsumptionAndComfortRewardFunction(RewardFunction):
    """
    Addition of `NetElectricityConsumptionReward` and `ComfortReward`.

    Parameters
    ----------
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any]
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any]
    :param exponent: Exponent to raise the temperature difference to if exceeding `comfort_band`.
    :type exponent: Optional[float] 
    :param coefficients: Coefficents for `NetElectricityConsumption` and `ComfortReward` values respectively.
    :type coefficients: Tuple[float], default (1.0, 1.0)
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], comfort_band: Optional[float]=None, exponent: Optional[float]=None, coefficients: Optional[Tuple[float]]=None):
        self.__functions: List[RewardFunction] = [
            NetElectricityConsumptionFunction(env_metadata=env_metadata),
            ComfortRewardFunction(env_metadata=env_metadata, comfort_band=comfort_band, exponent=exponent)
        ]
        super().__init__(env_metadata)
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @RewardFunction.env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        RewardFunction.env_metadata.fset(self, env_metadata)

        for f in self.__functions:
            f.env_metadata = self.env_metadata
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients

    def calculate(self, observations: Mapping[str, Union[int, float]]) -> float:
        # Compute each reward 
        reward = np.array([f.calculate(observations) for f in self.__functions], dtype=np.float32)
        
        # Scale rewards by coefficients and sum
        reward = reward*self.coefficients
        reward = reward.sum(dtype=np.float32).item()

        return reward
    

class ElectricityCostAndComfortRewardFunction(RewardFunction):
    """
    Addition of `ElectricityCostRewardFunction` and `ComfortReward`.

    Parameters
    ----------
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any]
    :param env_metadata: Static information about the environment.
    :type env_metadata: Mapping[str, Any]
    :param exponent: Exponent to raise the temperature difference to if exceeding `comfort_band`.
    :type exponent: Optional[float] 
    :param coefficients: Coefficents for `NetElectricityConsumption` and `ComfortReward` values respectively.
    :type coefficients: Tuple[float], default (1.0, 1.0)
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], comfort_band: Optional[float]=None, exponent: Optional[float]=None, coefficients: Optional[Tuple[float]]=None):
        self.__functions: List[RewardFunction] = [
            ElectricityCostRewardFunction(env_metadata=env_metadata),
            ComfortRewardFunction(env_metadata=env_metadata, comfort_band=comfort_band, exponent=exponent)
        ]
        super().__init__(env_metadata)
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @RewardFunction.env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        RewardFunction.env_metadata.fset(self, env_metadata)

        for f in self.__functions:
            f.env_metadata = self.env_metadata
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients

    def calculate(self, observations: Mapping[str, Union[int, float]]) -> float:
        # Compute each reward 
        reward = np.array([f.calculate(observations) for f in self.__functions], dtype=np.float32)
        
        # Scale rewards by coefficients and sum
        reward = reward*self.coefficients
        reward = reward.sum(dtype=np.float32).item()

        return reward