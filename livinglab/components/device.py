import numpy as np

from typing import Any, Optional, Mapping

from livinglab.base import Device


class ElectricDevice(Device):
    def __init__(self, efficiency: float, nominal_power: float, **kwargs: Mapping[str, Any]):
        super().__init__(efficiency=efficiency, **kwargs)
        self.nominal_power = nominal_power

    @property
    def nominal_power(self):
        return self._nominal_power
    
    @property
    def electricity_consumption(self):
        return self._electricity_consumption
    
    @property
    def available_nominal_power(self):
        return self._nominal_power - self._electricity_consumption[self.time_step]
    
    @nominal_power.setter
    def nominal_power(self, new_pow: float):
        assert new_pow >= 0, f'Invalid nominal power {new_pow}. Must be >= 0.'
        self._nominal_power = new_pow

    def update_electricity_consumption(self, electricity_consumption: float, enforce_polarity: bool=True):
        assert not enforce_polarity or electricity_consumption >= 0.0, \
            f'Invalid electricity consumption value {electricity_consumption}. Must be >= 0.'
        self._electricity_consumption[self.time_step] += electricity_consumption

    def reset(self):
        super().reset()
        self._electricity_consumption = np.zeros(self.episode_length, dtype=np.float32)


class HeatPump(ElectricDevice):
    def __init__(self, efficiency: float, nominal_power: float, mode: str, target_temperature: float, **kwargs: Mapping[str, Any]):
        super().__init__(efficiency=efficiency, nominal_power=nominal_power, **kwargs)
        self.mode = mode
        self.target_temperature = target_temperature

    @property
    def mode(self):
        return self._mode

    @property
    def target_temperature(self) -> float:
        return self._target_temperature
    
    @mode.setter
    def mode(self, new_mode: str):
        assert new_mode in ['heating', 'cooling'], f'Invalid Heat Pump mode {new_mode}. Must be either `heating` or `cooling`.'
        self._mode = new_mode

    @target_temperature.setter
    def target_temperature(self, new_temp: float):
        self._target_temperature = new_temp

    def get_cop(self, outdoor_dry_bulb_temperature: float) -> float:
        """
        Calculate the Carnot cycle CoP for heating or cooling mode. CoP is set to 20 if < 0 or > 20.

        Parameters
        ----------
        :param outdoor_dry_bulb_temperature: Outdoor dry bulb temperature [C].
        :type outdoor_dry_bulb_temperature: float

        Returns
        ----------
        :return: the calculated Coefficient of Performance.
        :rtype: float
        """
        # Convert temperature from Celcius fo Kelvin
        c_to_k = lambda x: x + 273.15
        outdoor_dry_bulb_temperature = np.array(outdoor_dry_bulb_temperature)

        # Calculate CoP
        if self._mode == 'heating':
            cop = self.efficiency*c_to_k(self._target_temperature)/(self._target_temperature - outdoor_dry_bulb_temperature)
        else:
            cop = self.efficiency*c_to_k(self._target_temperature)/(outdoor_dry_bulb_temperature - self._target_temperature)
        
        cop = np.array(cop)
        cop[cop < 0] = 20
        cop[cop > 20] = 20
        return cop

    def get_max_output_power(self, outdoor_dry_bulb_temperature: float, max_electric_power: Optional[float]) -> float:
        """
        Calculate maximum output power from heat pump given `cop`, `available_nominal_power` and `max_electric_power` limitations.

        Parameters
        ----------
        :param outdoor_dry_bulb_temperature: Outdoor dry bulb temperature [C].
        :type outdoor_dry_bulb_temperature: float
        :param max_electric_power: Maximum amount of electric power that the heat pump can consume from the power grid.
        :type max_electric_power: float

        Returns
        ----------
        :return: the calculated maximum output power.
        :rtype: float
        """
        # Compute CoP 
        cop = self.get_cop(outdoor_dry_bulb_temperature)

        if max_electric_power is None: 
            return self.available_nominal_power*cop  
        else:
            return np.min([max_electric_power, self.available_nominal_power], axis=0)*cop

    def get_input_power(self, output_power: float, outdoor_dry_bulb_temperature: float) -> float:
        """
        Calculate power needed to meet `output_power` given `cop` limitations.

        Parameters
        ----------
        :param output_power: Output power from heat pump.
        :type output_power: float
        :param outdoor_dry_bulb_temperature: Outdoor dry bulb temperature [C].
        :type outdoor_dry_bulb_temperature: float

        Returns
        ----------
        :return: the calculated input power to supply.
        :rtype: float
        """
        return output_power/self.get_cop(outdoor_dry_bulb_temperature)


class PVSystem:
    def __init__(self):
        pass