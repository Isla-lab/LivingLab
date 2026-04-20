# Omnisafe
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

# LivingLab
from livinglab.envs.livinglab_env import LivingLabEnv
from livinglab.utils.wrappers import NormalizedSpaceWrapper

# Utils
import torch
from typing import Any, ClassVar, Dict


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

        # Env info
        self._env = NormalizedSpaceWrapper(LivingLabEnv(**kwargs))
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space

        # KPI name to wandb log
        self.READABLE_KPIS = {
            'discomfort': 'Discomfort [%]',
            'indoor_dry_bulb_temperature_delta': 'Total Temperature Delta [°C]',
            'avg_indoor_dry_bulb_temperature_delta': 'Average Temperature Delta [°C]',
            'net_electricity_consumption': 'Total Electricity Consumption [kWh]',
            'avg_net_electricity_consumption': 'Average Electricity Consumption [kWh]',
            'net_electricity_consumption_cost': 'Total Electricity Consumption Cost [$]',
            'avg_net_electricity_consumption_cost': 'Average Electricity Consumption Cost [$]',
            'net_electricity_consumption_emissions': 'Total Electricity Consumption Emissions [kgCO2]',
            'avg_net_electricity_consumption_emissions': 'Average Electricity Consumption Emissions [kgCO2]'
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
        cost = torch.zeros_like(reward)

        return obs, reward, cost, terminated, truncated, info
    
    def set_seed(self, seed: int) -> None:
        self.reset(seed=seed)

    def render(self) -> Any:
        return self._env.render()

    def close(self) -> None:
        self._env.close()
