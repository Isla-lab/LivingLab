# Omnisafe
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

# LivingLab
from livinglab.envs.livinglab_env import LivingLabEnv
from livinglab.utils.wrappers import NormalizedSpaceWrapper

# Utils
import torch
import numpy as np
import pandas as pd
from typing import Any, Union, Callable, ClassVar, List, Dict


@env_register
class LivingLabOmnisafe(CMDP):
    _support_envs: ClassVar[list[str]] = ['LivingLab-v0']

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(
        self,
        env_id: str,
        device: torch.device=DEVICE_CPU,
        **kwargs
    ):
        super().__init__(env_id)
        
        # Required attrs
        self._num_envs = kwargs.pop('num_envs')
        self.type = 'LivingLab'

        # Cost function
        cost_cfgs = kwargs.pop('cost_fn')
        self._cost_fn = CostFunction(cost_cfgs['name'])
        self._cost_args = cost_cfgs.get('kwargs', {})

        # Env info
        self._env = NormalizedSpaceWrapper(LivingLabEnv(**kwargs))
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        self._cost_fn.env = self._env.unwrapped

        # KPI name to wandb log
        self.READABLE_KPIS = {
            'discomfort': 'Discomfort [%]',
            'min_indoor_dry_bulb_temperature_delta': 'Minimum Temperature Delta [°C]',
            'max_indoor_dry_bulb_temperature_delta': 'Maximum Temperature Delta [°C]',
            'avg_indoor_dry_bulb_temperature_delta': 'Average Temperature Delta [°C]',
            'ramping': 'Ramping [kWh]',
            'avg_daily_peak': 'Average Daily Peak [kWh]',
            'avg_global_peak': 'Average Global Peak [kWh]',
            'net_electricity_consumption': 'Total Electricity Consumption [kWh]',
            'net_electricity_consumption_cost': 'Total Electricity Consumption Cost [$]',
            'net_electricity_consumption_emissions': 'Total Electricity Consumption Emissions [kgCO2]',
        }

        # Device
        self._device = device

    @property
    def max_episode_steps(self):
        return self._env.unwrapped.episode_length - 1

    def reset(
        self,
        seed: int | None=None,
        options: Dict[str, Any] | None=None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        
        # Reset the wrapped environment
        obs, info = self._env.reset(seed=seed, options=options)
       
        # Convert observations to torch tensor
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info
    
    def step(self, action: torch.Tensor):

        # Convert actions to numpy
        action = action.detach().cpu().numpy()
        
        # Perform `.step()` in the wrapped env
        obs, reward, terminated, truncated, info = self._env.step(action)

        if terminated:
            # Get KPIs
            info = {
                self.READABLE_KPIS[k]: torch.as_tensor(v, dtype=torch.float32, device=self._device) 
                for k, v in info['kpis'].items() if k in self.READABLE_KPIS.keys()
            }
   
        # Retrieve `obs` and `reward` as torch tenors
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self._device)

        # Convert `truncated` and `terminated` into torch tensors
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self._device)
        truncated = torch.as_tensor(truncated, dtype=torch.bool, device=self._device)     

        # Placeholder
        cost = self._cost_fn(device=self._device, **self._cost_args)

        return obs, reward, cost, terminated, truncated, info
    
    def set_seed(self, seed: int) -> None:
        self.reset(seed=seed)

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()


class CostFunction:
    """
    Cost and flexibility functions to evaluate control performance.
    """

    def __init__(self, cost_fn: str, env: LivingLabEnv=None):
        self.env = env
        self.cost_fn = cost_fn

    def __call__(self, device: str | torch.device, **kwargs) -> torch.Tensor:
        if self._cost_fn is not None:
            cost =  self._cost_fn(**kwargs)
        else:
            cost = 0.0

        device = device if isinstance(device, torch.device) else torch.device(device)
        return torch.as_tensor(cost, dtype=torch.float32, device=device)

    @property
    def env(self) -> LivingLabEnv:
        return self._env
    
    @property
    def cost_fn(self) -> Callable:
        return self._cost_fn
    
    @env.setter
    def env(self, new_env: LivingLabEnv):
        assert new_env is None or isinstance(new_env, LivingLabEnv), f'Invalid environment class. Required `LivingLabEnv`, found {type(new_env)}.'
        self._env = new_env

    @cost_fn.setter
    def cost_fn(self, new_fn: str):
        self._cost_fn = getattr(self, new_fn)

    def ramping(self, exponent: float) -> float:
        """
        Compute the absolute difference in net electric consumption between consecutive time steps.

        Parameters
        ----------
        :param exponent: Exponent to raise the ramping to.
        :type exponent: float

        Returns
        ----------
        :return: Ramping cost time series.
        :rtype: float
        """

        prev_t = min(0, self.env.episode_time_step-2) 
        net_electricity_consumption = self.env.net_electricity_consumption
        ramping = abs(net_electricity_consumption[self.env.episode_time_step-1] - net_electricity_consumption[prev_t])**exponent
        
        return ramping

    def electricity_consumption(self, exponent: float) -> float:
        """
        Return the electricity consumption.

        Parameters
        ----------
        :param exponent: Exponent to raise the electricity consumption to.
        :type exponent: float
            
        Returns
        ----------
        :return: the net electricity consumption at `self.env.episode_time_step-1`
        :rtype: float
        """
        return max(0.0, self.env.net_electricity_consumption[self.env.episode_time_step-1])**exponent
    
    def cost(self, exponent: float) -> float:
        """
        Return the cost due to net electricity consumption.

        Parameters
        ----------
        :param exponent: Exponent to raise the electricity cost to.
        :type exponent: float
            
        Returns
        ----------
        :return: the cost of the electricity impoerted from the grid  at the current `self.env.episode_time_step-1`
        :rtype: float
        """
        return max(0.0, self.env.net_electricity_consumption_cost[self.env.episode_time_step-1])**exponent
    
    def carbon_emissions(self, exponent: float) -> float:
        """
        Return the emissions due to net electicity consumption.

        Parameters
        ----------
        :param exponent: Exponent to raise the carbon emissions to.
        :type exponent: float
            
        Returns
        ----------
        :return: the emissons due to net electricity consumption at the current `self.env.episode_time_step-1`
        :rtype: float
        """
        return max(0.0, self.env.net_electricity_consumption_emissions[self.env.episode_time_step-1])**exponent
    
    def discomfort(self, exponent: float) -> float:        
        """
        Return the absoulute difference between the indoor dry-bulb temperature and the setpoint.

        Parameters
        ----------
        :param exponent: exponent to raise the difference to if >= `comfort_band`
        :type exponent: float

        Returns
        ----------
        :return: the indoor temperature delta as cost
        :rtype: float
        """
        indoor_dry_bulb_temperature = self.env.energy_simulation.indoor_dry_bulb_temperature[self.env.time_step-1]
        indoor_dry_bulb_set_point = self.env.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[self.env.time_step-1]
        occupant_count = self.env.energy_simulation.occupant_count[self.env.time_step-1]
        comfort_band = self.env.energy_simulation.comfort_band[self.env.time_step-1]

        temp_delta = abs(indoor_dry_bulb_temperature - indoor_dry_bulb_set_point)
        cost = temp_delta**exponent if temp_delta >= comfort_band else temp_delta
        cost = 0.0 if occupant_count == 0 else cost

        return cost