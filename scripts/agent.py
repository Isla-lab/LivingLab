# LivingLab
from livinglab.envs.livinglab_env import LivingLabEnv

# Utils
import torch
import numpy as np
from torch.distributions import Normal
from typing import Any, Optional, Mapping


class OmnisafeActorWrapper:
    """
    Wrapper for loading and using an Omnisafe actor for evaluation.

    Parameters
    ----------
    :param fname: Path to the jit script of the actor's network.
    :type fname: str
    :param env: Living Lab environment to perform evaluation on.
    :type env: LivingLabEnv
    """

    def __init__(self, fname: str, env: LivingLabEnv):
        
        # Load actor net from file
        self.actor = torch.jit.load(fname)
        # CityLearn env action space for action scaling
        self.output_space = env.action_space

    def predict(self, obs: np.ndarray) -> np.ndarray:
        # Unwrap CityLearn observations
        obs = torch.as_tensor(obs, dtype=torch.float32)

        # Action distribution for given observation
        mean, log_std = self.actor(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        distr = Normal(mean, std)

        # Predicted action
        action = torch.tanh(distr.mean)
        action = action.detach().cpu().numpy()

        # Scale action to CityLearn range
        action = self._scale(action)

        return action
    
    def _scale(self, action: np.ndarray) -> np.ndarray:
        # Input space
        input_high = np.ones_like(self.output_space.high)
        input_low = -np.ones_like(self.output_space.low)
        input_range = input_high - input_low

        # Scale action to desired range
        output_range = self.output_space.high - self.output_space.low
        scaled_vector = (action - input_low) / input_range
        action = self.output_space.low + scaled_vector * output_range

        return action
    

class HourRBC:
    def __init__(self, env: LivingLabEnv):
        self.env_metadata = env.env_metadata
        
    def predict(self, observations: Mapping[str, Any]) -> np.ndarray:
        # Required information
        hour = observations['hour']
        t_air = observations['outdoor_dry_bulb_temperature']
        t_ground = observations['underground_temperature']
        mode = self.env_metadata['heat_pump']['mode']

        actions = []
        for action in self.env_metadata['action_names']:
            
            # Heat pump hourly rules
            if action == 'heat_pump':
                if 7 <= hour <= 15:
                    value = 0.7 if mode == 'cooling' else 0.3
                elif 16 <= hour <= 18:
                    value = 0.6 if mode == 'cooling' else 0.4
                elif 19 <= hour <= 22:
                    value = 0.8 if mode == 'cooling' else 0.6
                elif 23 <= hour <= 24:
                    value = 0.4 if mode == 'cooling' else 0.7
                elif 1 <= hour <= 6:
                    value = 0.2 if mode == 'cooling' else 0.8
                else:
                    value = 0.0

                # Greedly select source
                value = value if t_air <= t_ground else -value

            # Thermal battery hourly rules
            else:
                if 7 <= hour <= 15:
                    value = -0.02
                elif 16 <= hour <= 18:
                    value = -0.044
                elif 19 <= hour <= 22:
                    value = -0.024
                elif 23 <= hour <= 24:
                    value = 0.034
                elif 1 <= hour <= 6:
                    value = 0.05532
                else:
                    value = 0.0

            actions.append(value)

        return np.array(actions, dtype=np.float32)
    

class ComfortRBC(HourRBC):
    def __init__(self, env: LivingLabEnv, comfort_band: Optional[float]=None):
        super().__init__(env=env)
        self.comfort_band = comfort_band if comfort_band is not None else env.energy_simulation.comfort_band[0]

    def predict(self, observations: Mapping[str, Any]) -> np.ndarray:
        scheduled_actions = super().predict(observations).tolist()

        # Observations
        mode = self.env_metadata['heat_pump']['mode']
        outdoor_dry_bulb_temperature = observations['outdoor_dry_bulb_temperature']
        indoor_dry_bulb_temperature = observations['indoor_dry_bulb_temperature']
        indoor_temperature_set_point = observations[f'indoor_dry_bulb_temperature_{mode}_set_point']
        thm_soc = observations['thermal_battery_soc']

        # Action indexes 
        hp_idx = self.env_metadata['action_names'].index('heat_pump')
        thm_idx = self.env_metadata['action_names'].index('thermal_battery')

        if mode == 'cooling':
            hot_delta = indoor_dry_bulb_temperature - indoor_temperature_set_point
            
            # Above the set-point
            if hot_delta > 0:
                if hot_delta > self.comfort_band: # <- Too hot
                    scheduled_actions[hp_idx] = 0.8*np.sign(scheduled_actions[hp_idx])
                    if thm_soc > 0.1:
                        scheduled_actions[thm_idx] = min(scheduled_actions[thm_idx], -thm_soc/2)
                else: # <- Hot within the band
                    scheduled_actions[hp_idx] = 0.2*np.sign(scheduled_actions[hp_idx])
                    if thm_soc > 0.1:
                        scheduled_actions[thm_idx] = min(scheduled_actions[thm_idx], -thm_soc/3)

            # Below the set-point 
            else:
                temp_delta = outdoor_dry_bulb_temperature - indoor_dry_bulb_temperature
                if temp_delta > 0: # Outdoor temperature affects indoors
                    scheduled_actions[hp_idx] = 0.3*np.sign(scheduled_actions[hp_idx])
                else:
                    scheduled_actions[hp_idx] = 0.0

        return np.array(scheduled_actions, dtype=np.float32)