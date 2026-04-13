import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import os
from pathlib import Path
from typing import Any, Literal, Optional, Union, Callable, Iterable, Tuple, List, Mapping

from livinglab.base import Environment
from livinglab.components import HeatPump, PVSystem, ThermalBattery, LSTMDynamics
from livinglab.utils import EnergySimulation, Weather, Pricing, CarbonEmissions, PeriodicNormalization, ComfortRewardFuction


class LivingLabEnv(gym.Env, Environment):
    """
    LivingLab environment class.

    Parameters
    ----------
    :param seed: Experiment seed for reproducibility.
    :type seed: Optional[int]
    :param path: Path to the directory containing simulation data.
    :type path: Union[str, Path]
    :start_time_step: Simulation start time step.
    :type start_time_step: int
    :end_time_step: Simulation end time step.
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
            seed: Optional[int], 
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

        # Reward Function
        self.reward_fn = ComfortRewardFuction()

    @property
    def observation_names(self) -> List[str]:
        """Names of all observations that can be returned by the environment."""
        sim_data_names = self.energy_simulation.observation_names + self.weather.observation_names + self.pricing.observation_names + self.carbon_intensity.observation_names                
        device_obs_names = ['thermal_battery_soc', 'cooling_demand', 'net_electricity_consumption']

        return sim_data_names + device_obs_names
    
    @property
    def periodic_observations_metadata(self) -> Mapping[str, Any]:
        """Temporal periodic information observations."""
        return self._periodic_observations_metadata
    
    @property
    def terminated(self) -> bool:
        """Environment's termination signal. True when reaching the end of a simulation episode."""
        return self.episode_time_step >= (self.episode_length - 1)
    
    @property
    def truncated(self) -> Literal[False]:
        """Environment's truncation signal. Never used."""
        return False
    
    @property
    def info(self) -> Mapping[str, Any]:
        """Information dictionary returned upon calling `self.step()`."""
        _info = {}
        if self.terminated:
            _info['reward'] = {
                'min': self._episode_rewards.min(),
                'max': self._episode_rewards.max(),
                'sum': self._episode_rewards.sum(),
                'mean': self._episode_rewards.mean(),
            }

            # Devices' electricity consumption information
            heat_pump_electricity_consumption = self.heat_pump.electricity_consumption
            _info['heat_pump_electricity_consumption'] = {
                'min': heat_pump_electricity_consumption.min(),
                'max': heat_pump_electricity_consumption.max(),
                'sum': heat_pump_electricity_consumption.sum(),
                'mean': heat_pump_electricity_consumption.mean(),
            }

            thermal_battery_electricity_consumption = self.heat_pump.get_input_power(
                output_power=self.thermal_battery.energy_balance,
                outdoor_dry_bulb_temperature=self.weather.outdoor_dry_bulb_temperature[self.episode_start_time_step:self.episode_end_time_step+1]
            )
            _info['thermal_battery_electricity_consumption'] = {
                'min': thermal_battery_electricity_consumption.min(),
                'max': thermal_battery_electricity_consumption.max(),
                'sum': thermal_battery_electricity_consumption.sum(),
                'mean': thermal_battery_electricity_consumption.mean(),
            }


            # Total electricity consumption information
            _info['net_electricity_consumption'] = {
                'min': self._net_electricity_consumption.min(),
                'max': self._net_electricity_consumption.max(),
                'sum': self._net_electricity_consumption.sum(),
                'mean': self._net_electricity_consumption.mean(),
            }

            _info['net_electricity_consumption_cost'] = {
                'min': self._net_electricity_consumption_cost.min(),
                'max': self._net_electricity_consumption_cost.max(),
                'sum': self._net_electricity_consumption_cost.sum(),
                'mean': self._net_electricity_consumption_cost.mean(),
            }

            _info['net_electricity_consumption_emissions'] = {
                'min': self._net_electricity_consumption_emissions.min(),
                'max': self._net_electricity_consumption_emissions.max(),
                'sum': self._net_electricity_consumption_emissions.sum(),
                'mean': self._net_electricity_consumption_emissions.mean(),
            }

        return _info

    @property
    def observation_space(self) -> spaces.Box:
        """
        Environment's observation space.

        NOTE
        -----------
        Currently estimated from simulation data.
        """
        return self._observation_space
    
    @property
    def action_names(self) -> List[str]:
        """Names of actions that can be applied to the environment."""
        return ['heat_pump', 'thermal_battery']
    
    @property
    def action_space(self) -> spaces.Box:
        """Environment's action space."""
        return self._action_space
    
    @property
    def episode_rewards(self) -> List[float]:
        """List of rewards achieved within an episode."""
        return self._episode_rewards
    
    @property
    def net_electricity_consumption(self) -> List[float]:
        """Total electricity imported from the grid at each `episode_time_step` [kWh]."""
        return self._net_electricity_consumption
    
    @property
    def net_electricity_consumption_cost(self) -> List[float]:
        """Cost of the electricity imported from the grid at each `episode_time_step` [$*kWh]."""
        return self._net_electricity_consumption_cost
    
    @property
    def net_electricity_consumption_emissions(self) -> List[float]:
        """Emissions of the electricity imported from the grid at each `episode_time_step` [kgCO2*kWh]."""
        return self._net_electricity_consumption_emissions
    
    @property
    def simulate_dynamics(self) -> bool:
        """Signal for enabling dynamics simulation."""
        return self.dynamics._model_input[0][0] is not None

    @periodic_observations_metadata.setter
    def periodic_observations_metadata(self, new_metadata: Mapping[str, Iterable[Union[int, float]]]):
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
        self.pv_system.reset()

        # Reset dynamics
        self.dynamics.reset()

        # Reset additional variables
        self._episode_rewards = np.zeros(self.episode_length, dtype=np.float32)
        self._net_electricity_consumption = np.zeros(self.episode_length, dtype=np.float32)
        self._net_electricity_consumption_cost = np.zeros(self.episode_length, dtype=np.float32)
        self._net_electricity_consumption_emissions = np.zeros(self.episode_length, dtype=np.float32)

        return self.observations(names=options.get('names', False)), {}
    
    def step(self, actions: Union[np.ndarray | List[float]]) -> Tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
        """
        Run the environment's dynamics at the current `time_step`:
        1. apply `actions` to the controlled devices;
        2. update the building's dynamics to update the internal temperature;
        3. update internal variables to measure the effect of actions.

        Parameters
        ----------
        :param actions: Actions to apply to the devices.
        :type actions: Union[np.ndarray, List[float]]

        Returns
        ----------
        :return: a tuple containing next state observations, reward, termination signal and an info dictionary.
        :rtype: Tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]
        """
        assert len(actions) == self.action_space.shape[0] 
        if isinstance (actions, np.ndarray):
            actions = actions.tolist()

        # Pair actions to names
        action_dict = {}
        for name, action in zip(self.action_names, actions):
            action_dict[f'{name}_action'] = action

        # Apply actions to the environment
        self.apply_actions(**action_dict)

        # Update indoor temperature via dynamics
        self._update_dynamics_input()
        if self.simulate_dynamics:
            self.update_indoor_dry_bulb_temperature()

        # Update environment variables (reflect effects of actions)
        net_electricity_consumption = (
            self.energy_simulation.non_shiftable_load[self.time_step] +
            self.heat_pump.electricity_consumption[self.episode_time_step]
        ) - self.pv_system.get_generation(
                inverter_ac_power_per_kw=self.energy_simulation.solar_generation[self.time_step]
            )
        self._net_electricity_consumption[self.episode_time_step] = net_electricity_consumption
        self._net_electricity_consumption_cost[self.episode_time_step] = net_electricity_consumption * self.pricing.electricity_pricing[self.time_step] 
        self._net_electricity_consumption_emissions[self.episode_time_step] = net_electricity_consumption * self.carbon_intensity.carbon_intensity[self.time_step]

        # Compute reward
        reward_obs = self.observations(past=False, names=True)
        reward = self.reward_fn.calculate(observations=reward_obs)
        self._episode_rewards[self.episode_time_step] = reward

        # Advance to the next time step
        self._next_time_step()

        return self.observations(), reward, self.terminated, self.truncated, self.info

    def observations(self, periodic_normalization: Optional[bool]=None, past: bool=True, names: bool=False) -> Union[np.ndarray, Mapping[str, int|float]]:
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
        if periodic_normalization is None:
            periodic_normalization = self.periodic_normalization

        # Observations at current `self.time_step`
        data = self._get_observations_data(past=past)
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
    
    def apply_actions(self, heat_pump_action: float, thermal_battery_action: float):
        """
        Apply actions to the environment and simulate their impact by:
        - updating cooling/heating demand for the next time step;
        - charging/discharging the thermal battery.

        The order of execution depends on the polarity of the thermal battery action:
        - when discarching, `thermal_battery` is executed first;
        - when charging, `heat_pump` is executed first.

        This ensures that the discharged energy from the thermal battery is considered when allocating
        electricity for the heat pump to meet the LivingLab demand.

        Parameters
        ----------
        :param heat_pump_action: fraction of the Heat Pump `nominal_power` to make available.
        :type heat_pump_action: float
        :param thermal_battery_action: fraction of the Thermal Battery `capacity` to charge\discharge.
        :type thermal_battery_action: float
        """

        # Default action priority
        actions: Mapping[str, Tuple[Callable, Any]] = {
            'cooling_demand': (self.update_cooling_demand, (heat_pump_action,)),
            'heat_pump': (self.update_energy_from_heat_pump, ()),
            'thermal_battery': (self.update_thermal_battery, (thermal_battery_action,))
        }
        priority_list = list(actions.keys())

        # Check priority of `thermal_battery_action`
        if thermal_battery_action < 0.0:
            heat_pump_idx = priority_list.index('heat_pump')
            thermal_battery_idx = priority_list.index('thermal_battery')
            priority_list[heat_pump_idx] = 'thermal_battery'
            priority_list[thermal_battery_idx] = 'heat_pump'

        # Apply actions according to the priority
        for key in priority_list:
            func, args = actions[key]
            func(*args)
    
    def estimate_observation_space(self, periodic_normalization: Optional[bool]=None) -> spaces.Box:
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
        if periodic_normalization is None:
            periodic_normalization = self.periodic_normalization
            
        # Get observation space limits
        low, high = self.estimate_observation_space_limits(periodic_normalization=periodic_normalization)
        low, high = list(low.values()), list(high.values())

        return spaces.Box(low=np.array(low, dtype=np.float32), high=np.array(high, dtype=np.float32), dtype=np.float32)

    def estimate_observation_space_limits(self, periodic_normalization: Optional[bool]=None) -> Tuple[Mapping[str, float], Mapping[str, float]]:
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
        if periodic_normalization is None:
            periodic_normalization = self.periodic_normalization

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
                high[key] = sim_data[key].max()

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
    
    def update_cooling_demand(self, heat_pump_action: float):
        """
        Update the space cooling demand for both the current `time_step`.

        Parameters
        ----------
        :param heat_pump_action: fraction of the Heat Pump `nominal_power` made available for space cooling.
        :type heat_pump_action: float
        """
        # Calculate cooling demand according to the heat pump action
        if self.simulate_dynamics:
            output_power = heat_pump_action * self.heat_pump.nominal_power
            demand = self.heat_pump.get_max_output_power(
                outdoor_dry_bulb_temperature=self.weather.outdoor_dry_bulb_temperature[self.time_step],
                max_electric_power=output_power
            )

            # Update demand for both the current and the next time step
            self.energy_simulation.cooling_demand[self.time_step] = demand

    def update_energy_from_heat_pump(self):
        """
        Update Heat Pump electricity consumption given the current `time_step` cooling demand.
        """
        # Retreive current observations
        demand = self.energy_simulation.cooling_demand[self.time_step]
        temperature = self.weather.outdoor_dry_bulb_temperature[self.time_step]
        thermal_battery_output = min(self.thermal_battery.energy_balance[self.episode_time_step], 0.0)

        # Maximum possible heat pump output
        max_hp_output = self.heat_pump.get_max_output_power(outdoor_dry_bulb_temperature=temperature, max_electric_power=None)
        assert demand <= max_hp_output, f'[STEP:{self.time_step}] Cooling demand exceeds Heat Pump maximum output ' + \
            f'(demand={demand} > max_output={max_hp_output})'

        # Actual Heat Pump output and consumption
        heat_pump_output = min(demand - thermal_battery_output, max_hp_output)
        electricity_consumption = self.heat_pump.get_input_power(output_power=heat_pump_output, outdoor_dry_bulb_temperature=temperature)
        assert electricity_consumption >= 0.0, f'[STEP: {self.time_step}] Negative electricity consumption for cooling demand. Found {electricity_consumption}'

        # Update
        self.heat_pump.update_electricity_consumption(electricity_consumption)

    def update_thermal_battery(self, thermal_battery_action: float):
        """
        Charge/Discharge the Thermal Battery for the current `time_step`.

        Parameters
        ----------
        :param thermal_battery_action: fraction of the Thermal Battery `capacity` to charge\discharge.
        :type thermal_battery_action: float
        """
        energy = thermal_battery_action * self.thermal_battery.capacity
        temerature = self.weather.outdoor_dry_bulb_temperature[self.time_step]

        # Charge the Thermal Battery via the Heat Pump
        if energy > 0.0:
            max_hp_output = self.heat_pump.get_max_output_power(outdoor_dry_bulb_temperature=temerature, max_electric_power=None)
            energy = min(energy, max_hp_output)

        # Supply the cooling demand via the Thermal Battery first
        else:
            demand = self.energy_simulation.cooling_demand[self.time_step]
            energy = max(energy, -demand)

        # Update the battery status
        self.thermal_battery.charge(energy)

        # Compute the required electricity to charge the battery
        charged_energy = max(self.thermal_battery.energy_balance[self.episode_time_step], 0.0)
        electricity_consumption = self.heat_pump.get_input_power(output_power=charged_energy, outdoor_dry_bulb_temperature=temerature)
        self.heat_pump.update_electricity_consumption(electricity_consumption)

    def update_indoor_dry_bulb_temperature(self):
        """
        Predict and update the indoor temperature for the current `time_step`.
        """
        # Predict
        input_tensor = self._get_dynamics_input()
        hidden_state = tuple([h.data for h in self.dynamics.hidden_state])
        indoor_dry_bulb_temperature_norm, self.dynamics.hidden_state = self.dynamics(input_tensor, h=hidden_state)

        # Update indoor dry bulb temperature in the model's input
        idx = self.dynamics.input_observation_names.index('indoor_dry_bulb_temperature')
        self.dynamics.model_input[idx][-1] = indoor_dry_bulb_temperature_norm.item() 

        # Unormalize and update indoor dry bulb temperature in simulation for the current time step
        min_, max_ = self.dynamics.input_norm_min[idx], self.dynamics.input_norm_max[idx]
        indoor_dry_bulb_temperature = indoor_dry_bulb_temperature_norm*(max_ - min_) + min_
        self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = indoor_dry_bulb_temperature.item()

    def _get_observations_data(self, past: bool=True) -> Mapping[str, Union[int, float]]:        
        # Current simulation data
        observations = {
            **self.energy_simulation.observations(time_step=self.time_step),
            **self.weather.observations(time_step=self.time_step),
            **self.pricing.observations(time_step=self.time_step),
            **self.carbon_intensity.observations(time_step=self.time_step)
        }

        # Update solar generation
        observations['solar_generation'] = self.pv_system.get_generation(inverter_ac_power_per_kw=observations['solar_generation'])

        # Update with possible past observations
        past_t = min(self.time_step - 1, 0) if past else self.time_step
        ep_past_t = min(self.episode_time_step - 1, 0) if past else self.episode_time_step
        observations.update({
            'cooling_demand': self.energy_simulation.cooling_demand[past_t] + abs(min(self.thermal_battery.energy_balance[ep_past_t], 0.0)),
            'thermal_battery_soc': self.thermal_battery.soc[ep_past_t],
            'net_electricity_consumption': self.net_electricity_consumption[ep_past_t]
        })

        return observations
    
    def _next_time_step(self):
        # Advance all components
        self.heat_pump.step()
        self.thermal_battery.step()
        self.pv_system.step()

        # Increate `self.time_step`
        Environment.step(self)
    
    def _update_dynamics_input(self, sanity_check: bool=True):
        # Get current observations
        obs = self.observations(past=False, names=True)
        if sanity_check:
            missing = [name for name in self.dynamics.input_observation_names if name not in obs.keys()]
            assert len(missing) == 0, f'Missing observations required by the dynamics: {missing}.'

        # Append current observations to the model's input
        self.dynamics.model_input = [
            l[-self.dynamics.lookback:] + [(obs[k] - min_)/(max_ - min_)]
            for l, k, min_, max_ in zip(
                self.dynamics.model_input,
                self.dynamics.input_observation_names,
                self.dynamics.input_norm_min,
                self.dynamics.input_norm_max
            )
        ]

    def _get_dynamics_input(self) -> torch.Tensor:
        model_input = []

        # Collect observations from the previous time steps
        for i, k in enumerate(self.dynamics.input_observation_names):
            if k == 'indoor_dry_bulb_temperature':
                model_input.append(self.dynamics.model_input[i][:-1])
            else:
                model_input.append(self.dynamics.model_input[i][1:])

        # Create torch tensor
        model_input = torch.tensor(model_input, dtype=torch.float32)
        model_input = model_input.T.unsqueeze(0)
        return model_input