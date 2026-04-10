import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Mapping

from livinglab.base import Environment
from livinglab.components import HeatPump, PVSystem, ThermalBattery, LSTMDynamics
from livinglab.utils import EnergySimulation, Weather, Pricing, CarbonEmissions, PeriodicNormalization


class LivingLabEnv(gym.Env, Environment):
    """
    LivingLab environment class.

    Parameters
    ----------
    :param seed: Experiment seed for reproducibility.
    :type seed: int
    :param path: Path to the directory containing simulation data.
    :type path: Union[str, Path]
    :param start_time_step:
    :type start_time_step: int
    :param end_time_step:
    :type end_time_step: int
    :param episode_length: Episode duration in time steps.
    :type episode_length: int
    :param heat_pump_cfgs: Heat Pump configurations.
    :type heat_pump_cfgs: Mapping[str, Any]
    :param thermal_battery_cfgs: Thermal Battery configurations.
    :type thermal_battery_cfgs: Mapping[str, Any]
    :param pv_system_cfgs: PV System configurations.
    :type pv_system_cfgs: Mapping[str, Any]
    :param dynamics_cfgs: Environment dynamics configurations.
    :type dynamics_cfgs: Mapping[str, Any]
    :param periodic_normalization: Whether to normalize periodic temporal observations.
    :type periodic_normalization: bool
    """
    
    def __init__(
            self,
            seed: int, 
            path: Union[str, Path],
            start_time_step: int,
            end_time_step: int,
            heat_pump_cfgs: Mapping[str, Any],
            thermal_battery_cfgs: Mapping[str, Any],
            pv_system_cfgs: Mapping[str, Any],
            dynamics_cfgs: Mapping[str, Any],
            periodic_normalization: bool,
            episode_length: Optional[int]=None,
        ):
        super().__init__(seed=seed, start_time_step=start_time_step, end_time_step=end_time_step, episode_length=episode_length)

        # Options
        self.periodic_normalization = periodic_normalization

        # Simulation data
        self.energy_simulation = EnergySimulation(path=os.path.join(path, 'Building_1.csv'), start_time_step=start_time_step, end_time_step=end_time_step)
        self.weather = Weather(path=os.path.join(path, 'weather.csv'), start_time_step=start_time_step, end_time_step=end_time_step)
        self.pricing = Pricing(path=os.path.join(path, 'pricing.csv'), start_time_step=start_time_step, end_time_step=end_time_step)
        self.carbon_intensity = CarbonEmissions(path=os.path.join(path, 'carbon_intensity.csv'), start_time_step=start_time_step, end_time_step=end_time_step)

        # Devices
        self.heat_pump = HeatPump(**heat_pump_cfgs, seed=seed, start_time_step=start_time_step, end_time_step=end_time_step, episode_length=episode_length)
        self.thermal_battery = ThermalBattery(**thermal_battery_cfgs, seed=seed, start_time_step=start_time_step, end_time_step=end_time_step, episode_length=episode_length)
        self.pv_system = PVSystem(**pv_system_cfgs, seed=seed, start_time_step=start_time_step, end_time_step=end_time_step, episode_length=episode_length)

        # Dynamics
        self.dynamics = LSTMDynamics(**dynamics_cfgs)

        # Observation/action spaces
        self.periodic_observations_metadata = {'hour': range(1, 25), 'day_type': range(1, 8), 'month': range(1, 13)}
        self.observation_space = self.estimate_observation_space(periodic_normalization=periodic_normalization)
        self.action_space = self.estimate_action_space()

    @property
    def observation_names(self):
        sim_data_names = self.energy_simulation.observation_names + self.weather.observation_names + self.pricing.observation_names + self.carbon_intensity.observation_names                
        device_obs_names = ['thermal_battery_soc', 'cooling_demand', 'net_electricity_consumption']

        return sim_data_names + device_obs_names
    
    @property
    def periodic_observations_metadata(self):
        return self._periodic_observations_metadata
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def action_names(self):
        return ['heat_pump', 'thermal_battery']
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def net_electricity_consumption(self):
        return self._net_electricity_consumption

    @periodic_observations_metadata.setter
    def periodic_observations_metadata(self, new_metadata: Mapping[str, Tuple[Union[int, float], Union[int, float]]]):
        self._periodic_observations_metadata = dict(**new_metadata)
    
    @observation_space.setter
    def observation_space(self, new_space: spaces.Box):
        self._observation_space = new_space

    @action_space.setter
    def action_space(self, new_space: spaces.Box):
        self._action_space = new_space

    def reset(self, seed: int=None, options: Mapping[str, Any]={}) -> Tuple[np.ndarray, Mapping[str, Any]]:
        """
        Reset `LivingLabEnv` to its initial state.

        Parameters
        ----------
        :param seed: use to update `envs.livinglab_env.LivingLabEnv` random seed if provided.
        :type seed: int
        :param options: Not used. Included to conform to gymnasium interface.
        :type options: Mapping[str, Any]

        Returns
        ----------
        :return: the initial state observations
        :rtype: np.ndarray
        """
        gym.Env.reset(self)
        Environment.reset(self)

        # Update seed
        if seed is not None:
            self.seed = seed

        # Reset devices
        self.heat_pump.reset()
        self.thermal_battery.reset()

        # Reset dynamics
        self.dynamics.reset()

        # Reset additional variables
        self._net_electricity_consumption = np.zeros(self.episode_length, dtype=np.float32)

        return self.observations(periodic_normalization=self.periodic_normalization, names=options.get('names', False)), {}

    def observations(self, periodic_normalization: bool=False, names: bool=False) -> Union[np.ndarray, Mapping[str, int|float]]:
        """
        Return observations at the current `self.time_step`.

        Parameters
        ----------
        :param periodic_normalization: whether to normalize periodic temporal variables
        :type periodic_normalization: bool
        :param names: whether to return a dictionary with observation names
        :type names: bool

        Returns
        ----------
        :return: the current observation
        :rtype: Union[np.ndarray, Mapping[str, int|float]]
        """
        # Observations at current `self.time_step`
        data = self._get_observations_data()
        observations = {}

        # Periodic observations normalization
        if periodic_normalization:
            pn = PeriodicNormalization(x_max=0)
            periodic_observations = self.periodic_observations_metadata
            for k, v in data.items():
                if k in periodic_observations:
                    pn.x_max = max(periodic_observations[k])
                    sin_x, cos_x = (pn * v).astype(dtype=np.float32)
                    observations[f'{k}_sin'] = sin_x
                    observations[f'{k}_cos'] = cos_x
                else:
                    observations[k] = v
        else:
            observations = dict(**data)

        # Return dictionary with observation names
        if names:
            return observations
        
        return np.array(list(observations.values()), dtype=np.float32)
    
    def estimate_observation_space(self, periodic_normalization: bool=False) -> spaces.Box:
        """
        Estimate observation space from simulation data.

        Parameters
        ----------
        :param periodic_normalization: whether to normalize periodic temporal variables
        :type periodic_normalization: bool

        Returns
        ----------
        :return: the estimated observation space
        :rtype: gym.spaces.Box
        """
        # Get observation space limits
        low, high = self.estimate_observation_space_limits(periodic_normalization=periodic_normalization)
        low, high = list(low.values()), list(high.values())

        return spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)

    def estimate_observation_space_limits(self, periodic_normalization: bool=False) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        """
        Estimate observation space limits for each variable from simulation data.

        Parameters
        ----------
        :param periodic_normalization: whether to normalize periodic temporal variables
        :type periodic_normalization: bool

        Returns
        ----------
        :return: the estimated observation space limits
        :rtype: Tuple[Mapping[str, float], Mapping[str, float]]
        """
        # Get simulation data to estimate space limits
        sim_data = {
            **self.energy_simulation.observations(),
            **self.weather.observations(),
            **self.pricing.observations(),
            **self.carbon_intensity.observations(),
        }

        # Estimate space limits
        low, high = {}, {}
        for key in self.observation_names:
            if key == 'solar_generation':
                low[key] = self.pv_system.get_generation(inverter_ac_power_per_kw=sim_data['solar_generation'].min())
                high[key] = self.pv_system.get_generation(inverter_ac_power_per_kw=sim_data['solar_generation'].max())

            elif key == 'comfort_band':
                low[key] = 0.0
                high[key] = sim_data[key].max() 

            elif key == 'thermal_battery_soc':
                low[key] = 0.0
                high[key] = 1.0

            elif key == 'net_electricity_consumption':
                low[key] = -self.pv_system.get_generation(inverter_ac_power_per_kw=sim_data['solar_generation'].max())
                high[key] = sim_data['non_shiftable_load'].max() + self.heat_pump.nominal_power

            elif key == 'cooling_demand':
                low[key] = 0.0
                high[key] = self.heat_pump.nominal_power

            elif key in self.periodic_observations_metadata.keys():
                periodic_observations = self.periodic_observations_metadata[key]
                if periodic_normalization:
                    pn = PeriodicNormalization(max(periodic_observations))
                    x_sin, x_cos = pn * np.array(periodic_observations)
                    low[f'{key}_sin'], high[f'{key}_sin'] = x_sin.min(), x_sin.max()
                    low[f'{key}_cos'], high[f'{key}_cos'] = x_cos.min(), x_cos.max()
                else:
                    low[key] = min(periodic_observations)
                    high[key] = max(periodic_observations)

            else:
                low[key] = sim_data[key].min()
                high[key] = sim_data[key].max()

        return low, high
    
    def estimate_action_space(self) -> spaces.Box:
        """
        Estimate action space from devices' specifications.

        Returns
        ----------
        :return: the estimated action space
        :rtype: gym.spaces.Box
        """
        low, high = [], []

        for key in self.action_names:
            if key == 'heat_pump':
                low.append(0.0)
                high.append(1.0)

            elif key == 'thermal_battery':
                limit = self.heat_pump.nominal_power/max(self.thermal_battery.capacity, 1e-6)
                limit = min(limit, 1.0)
                low.append(-limit)
                high.append(limit)

        return spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)

    def _get_observations_data(self) -> Mapping[str, Union[int, float]]:        
        observations = {
            # Simulation data
            **self.energy_simulation.observations(time_step=self.time_step),
            **self.weather.observations(time_step=self.time_step),
            **self.pricing.observations(time_step=self.time_step),
            **self.carbon_intensity.observations(time_step=self.time_step),
            # Devices info
            'thermal_battery_soc': self.thermal_battery.soc[self.time_step],
            'cooling_demand': self.heat_pump.electricity_consumption[self.time_step], # <- Total Heat Pump electricity consumption
            'net_electricity_consumption': self.net_electricity_consumption[self.time_step]
        }

        # Update solar generation
        observations['solar_generation'] = self.pv_system.get_generation(inverter_ac_power_per_kw=observations['solar_generation'])

        return observations 