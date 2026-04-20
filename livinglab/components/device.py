import numpy as np

from typing import Any, Optional, Mapping

from livinglab.base import Device


class ElectricDevice(Device):
    """
    Electric device class.

    Parameters
    ----------
    :param efficiency: Technical efficiency.
    :type efficiency: float
    :param nominal_power: Device's nominal power.
    :type nominal_power: float
    :param **kwargs: Other keyword arguments to initialize super classes.
    :type **kwargs: Mapping[str, Any]
    """
    def __init__(self, efficiency: Optional[float], nominal_power: float, **kwargs: Mapping[str, Any]):
        super().__init__(efficiency=efficiency, **kwargs)
        self.nominal_power = nominal_power

    @property
    def nominal_power(self) -> float:
        """Electrical Device's nominal power [kWh]."""
        return self._nominal_power
    
    @property
    def electricity_consumption(self) -> np.ndarray:
        """Evolution of the electricity consumed by the device within a simulation episode [kWh]."""
        return self._electricity_consumption
    
    @property
    def available_nominal_power(self):
        return self._nominal_power - self._electricity_consumption[self.episode_time_step]
    
    @nominal_power.setter
    def nominal_power(self, new_pow: float):
        assert new_pow >= 0, f'Invalid nominal power {new_pow}. Must be >= 0.'
        self._nominal_power = new_pow

    def update_electricity_consumption(self, electricity_consumption: float, enforce_polarity: bool=True):
        """
        Update the electricity consumed by the device at the current `episode_time_step`.

        Parameters
        ----------
        :param electricity_consumption: Consumed electricity.
        :type electricity_consumption: float
        :param enforce_polarity: Whether to consider only positive consumptions.
        :type enforce_polarity: bool
        """
        assert not enforce_polarity or electricity_consumption >= 0.0, \
            f'Invalid electricity consumption value {electricity_consumption}. Must be >= 0.'
        self._electricity_consumption[self.episode_time_step] += electricity_consumption

    def reset(self):
        """Reset the Electric Device to its initial state."""
        super().reset()
        self._electricity_consumption = np.zeros(self.episode_length, dtype=np.float32)


class HeatPump(ElectricDevice):
    """
    Electric device class.

    Parameters
    ----------
    :param efficiency: Technical efficiency.
    :type efficiency: float
    :param nominal_power: Heat Pump's nominal power.
    :type nominal_power: float
    :param mode: Heat Pump HVAC mode (either `cooling` or `heating`).
    :type mode: str
    :param target_temperature: Target temperature for CoP measurement.
    :type target_temperature: float
    :param **kwargs: Other keyword arguments to initialize super classes.
    :type **kwargs: Mapping[str, Any]
    """
    def __init__(self, efficiency: float, nominal_power: float, mode: str, target_temperature: float, **kwargs: Mapping[str, Any]):
        super().__init__(efficiency=efficiency, nominal_power=nominal_power, **kwargs)
        self.mode = mode
        self.target_temperature = target_temperature

    @property
    def mode(self) -> str:
        """Heat Pump HVAC mode."""
        return self._mode

    @property
    def target_temperature(self) -> float:
        """Heat Pump target temperature for CoP measurement."""
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
        outdoor_dry_bulb_temperature = np.array(outdoor_dry_bulb_temperature, dtype=np.float32)

        # Calculate CoP
        if self._mode == 'heating':
            cop = self.efficiency*c_to_k(self._target_temperature)/(self._target_temperature - outdoor_dry_bulb_temperature)
        else:
            cop = self.efficiency*c_to_k(self._target_temperature)/(outdoor_dry_bulb_temperature - self._target_temperature)
        
        cop = np.array(cop, dtype=np.float32)
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
            max_out_power = self.available_nominal_power*cop  
        else:
            max_out_power = np.min([max_electric_power, self.available_nominal_power], axis=0)*cop

        return max_out_power.astype(np.float32)

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


class PVSystem(ElectricDevice):
    """
    Base PV system class.

    Parameters
    ----------
    :param nominal_power: PV output power [kW].
    :type nominal_power: float
    :param **kwargs: Keyword arguments to initialize super class.
    :type **kwargs: Mapping[str, Any]
    """
    def __init__(self, nominal_power: float, **kwargs: Mapping[str, Any]):
        super().__init__(efficiency=None, nominal_power=nominal_power, **kwargs)

    def get_generation(self, inverter_ac_power_per_kw: float) -> float:
        """
        Get solar generation output.

        Parameters
        ----------
        :param inverter_ac_power_per_kw: Inverter AC power per kW of PV nominal power [W/kW]
        :type inverter_ac_power_per_kw: float

        Returns
        ----------
        :return: the solar generation output.
        :rtype: float
        """
        return self.nominal_power*inverter_ac_power_per_kw/1000.0