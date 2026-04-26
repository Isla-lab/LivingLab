import numpy as np
import pandas as pd
from ladybug.epw import EPW

from abc import ABC, abstractmethod
from typing import Optional, Mapping, Union, Iterable, List

from livinglab.utils.functions import sol_air_temperature


class TimeSeriesData(ABC):
    """
    Generic time series data class.    
    
    Parameters
    ----------
    :param path: Path to time series DataFrame.
    :param start_time_step: Time step to start reading variables.
    :type start_time_step: int, optional
    :param end_time_step: Time step to end reading variables.
    :type end_time_step: int, optional
    :param noise_std: Gaussian noise standard deviation
    :type noise_std: float
    """

    def __init__(self, path: str, start_time_step: Optional[int], end_time_step: Optional[int], noise_std: float=0.0):        
        # Simulation window
        self.start_time_step = start_time_step
        self.end_time_step = end_time_step

        self.noise_std = noise_std

        # Process simulation DataFrame
        sim_df = pd.read_csv(path, sep=',')
        self._load(df=sim_df)

    @abstractmethod
    def _load(self, df: pd.DataFrame):
        """
        Load data from the simulation DataFrame.

        Parameters
        ----------
        :param df: Simulation DataFrame.
        :type df: pd.DataFrame
        """
        pass

    @property
    def observation_names(self) -> List[str]:
        """List of time series variable names."""
        return [k.lstrip('_') for k, v in vars(self).items() if isinstance(v, np.ndarray)]
    
    def observations(self, time_step: int=None) -> Mapping[str, Union[int, float, np.ndarray]]:
        """
        Return the full time series observations or at a given `time_step`.

        Parameters
        ----------
        :param time_step: Time step to observe time series variables (default is `None`).
        :type time_step: int

        Returns
        ----------
        :return: the mapping of time series observations in the form of {variate: observation}
        :rtype: Mapping[str, Union[int, float, np.ndarray]]
        """
        full_obs = {k.lstrip('_'): v for k, v in vars(self).items() if isinstance(v, np.ndarray)}

        if time_step is not None:
            assert self.start_time_step <= time_step <= self.end_time_step
            return {k: v[time_step] for k, v in full_obs.items()}
        
        return full_obs

    def add_gaussian_noise(self, variable: Union[Iterable[float], np.ndarray]) -> np.ndarray:
        """
        Add Gaussian noise to a time series variable.

        Parameters
        ----------
        :param variable: Time series variable to add noise to.
        :type variable: Union[Iterable[float], np.ndarray]

        Returns
        ----------
        :return: the noisy variable if `self.noise_std > 0`, unchanged otherwise
        :rtype: np.ndarray
        """
        variable = np.asarray(variable, dtype=np.float32)

        if self.noise_std <= 0.0:
            return variable
        
        # Add noise given `self.noise_std`
        noise = np.random.normal(loc=0, scale=self.noise_std, size=variable.shape)
        return variable + noise


class EnergySimulation(TimeSeriesData):
    """
    Living Lab energy simulation.
    
    Parameters
    ----------
    :param path: Path to time series DataFrame.
    :param start_time_step: Time step to start reading variables.
    :type start_time_step: int, optional
    :param end_time_step: Time step to end reading variables.
    :type end_time_step: int, optional
    :param noise_std: Gaussian noise standard deviation
    :type noise_std: float
    :param comfort_band: Temperature comfort band with respect to the set point.
    :type comfort_band: float
    """

    def __init__(self, path: str, start_time_step: Optional[int], end_time_step: Optional[int], noise_std: float=0.0, comfort_band: float=2.0):
        super().__init__(path=path, start_time_step=start_time_step, end_time_step=end_time_step, noise_std=noise_std)
        self.comfort_band = np.zeros(self.hour.shape[0], dtype=np.float32) + comfort_band

    def _load(self, df: pd.DataFrame):
        sim_data = df.to_dict('list')

        # Temporal data
        self.month = np.array(sim_data['month'], dtype=np.int32)
        self.hour = np.array(sim_data['hour'], dtype=np.int32)
        self.day_type = np.array(sim_data['day_type'], dtype=np.int32)

        # Ambient's information
        self.indoor_dry_bulb_temperature = np.clip(self.add_gaussian_noise(sim_data['indoor_dry_bulb_temperature']), -90, 57)
        self.indoor_relative_humidity = np.clip(self.add_gaussian_noise(sim_data['indoor_relative_humidity']), 0, 100)
        self.occupant_count = np.array(sim_data['occupant_count'], dtype=np.float32)
        self.indoor_dry_bulb_temperature_cooling_set_point = np.array(sim_data['indoor_dry_bulb_temperature_cooling_set_point'], dtype=np.float32)

        # Additional energy demands
        self.cooling_demand = np.array(sim_data['cooling_demand'], dtype=np.float32)
        self.non_shiftable_load = np.array(sim_data['non_shiftable_load'], dtype=np.float32)
        self.solar_generation = self.add_gaussian_noise(sim_data['solar_generation'])

class Weather(TimeSeriesData):
    """
    Outdoor weather simulation.
    
    Parameters
    ----------
    :param path: Path to time series DataFrame.
    :type path: str
    :param start_time_step: Time step to start reading variables.
    :type start_time_step: int, optional
    :param end_time_step: Time step to end reading variables.
    :type end_time_step: int, optional
    :param noise_std: Gaussian noise standard deviation
    :type noise_std: float
    """

    def __init__(self, path: str, start_time_step: Optional[int], end_time_step: Optional[int], noise_std: float=0.0):
        super().__init__(path=path, start_time_step=start_time_step, end_time_step=end_time_step, noise_std=noise_std)

    def _load(self, df: pd.DataFrame):
        sim_data = df.to_dict('list')

        # Weather data
        self.outdoor_dry_bulb_temperature = self.add_gaussian_noise(sim_data['outdoor_dry_bulb_temperature'])
        self.outdoor_relative_humidity = self.add_gaussian_noise(sim_data['outdoor_relative_humidity'])
        self.diffuse_solar_irradiance = self.add_gaussian_noise(sim_data['diffuse_solar_irradiance'])
        self.direct_solar_irradiance = self.add_gaussian_noise(sim_data['direct_solar_irradiance'])

        # Surface level and underground temperature calculation
        self.surface_ground_temperature = sol_air_temperature(
            t_air=self.outdoor_dry_bulb_temperature,
            g_direct=self.direct_solar_irradiance,
            g_diffuse=self.diffuse_solar_irradiance
        )


class Pricing(TimeSeriesData):
    """
    Electricity pricing simulation based on a given rate ($/kWh).
    
    Parameters
    ----------
    :param path: Path to time series DataFrame.
    :param start_time_step: Time step to start reading variables.
    :type start_time_step: int, optional
    :param end_time_step: Time step to end reading variables.
    :type end_time_step: int, optional
    :param noise_std: Gaussian noise standard deviation
    :type noise_std: float
    """

    def __init__(self, path: str, start_time_step: Optional[int], end_time_step: Optional[int], noise_std: float=0.0):
        super().__init__(path=path, start_time_step=start_time_step, end_time_step=end_time_step, noise_std=noise_std)

    def _load(self, df: pd.DataFrame):
        sim_data = df.to_dict('list')

        # Current electricity pricing
        self.electricity_pricing = np.array(sim_data['electricity_pricing'], dtype=np.float32)


class CarbonEmissions(TimeSeriesData):
    """
    Carbon emissions due to electricity import (KgCO2/kWh).
    
    Parameters
    ----------
    :param path: Path to time series DataFrame.
    :param start_time_step: Time step to start reading variables.
    :type start_time_step: int, optional
    :param end_time_step: Time step to end reading variables.
    :type end_time_step: int, optional
    :param noise_std: Gaussian noise standard deviation
    :type noise_std: float
    """

    def __init__(self, path: str, start_time_step: Optional[int], end_time_step: Optional[int], noise_std: float=0.0):
        super().__init__(path=path, start_time_step=start_time_step, end_time_step=end_time_step, noise_std=noise_std)

    def _load(self, df: pd.DataFrame):
        sim_data = df.to_dict('list')

        # Current carbon emissions
        self.carbon_intensity = np.array(sim_data['carbon_intensity'], dtype=np.float32)
