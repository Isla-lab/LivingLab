import numpy as np

from typing import Any, Optional, Mapping

from livinglab.base import Device


class ThermalBattery(Device):
    """
    Thermal Battery class.

    Parameters
    ----------
    :param efficiency: Technical efficiency.
    :type efficiency: float
    :param capacity: Maximum battery capacity.
    :type capacity: float
    :param loss_coef: Capacity loss between consecutive time steps.
    :type loss_coef: float
    :param initial_soc: Battery State of Charge at the beginning of simulation.
    :type initial_soc: Optional[float]
    :param max_input_power: Maximum amount of energy that can be input to the battery at each time step.
    :type max_input_power: Optional[float]
    :param max_output_power: Maximum amount of energy that can be output by the battery at each time step.
    :type max_output_power: Optional[float]
    :param **kwargs: Other keyword arguments to initialize super classes.
    :type **kwargs: Mapping[str, Any]
    """
    def __init__(
            self, 
            efficiency: float, 
            capacity: float, 
            loss_coef: float, 
            initial_soc: Optional[float]=None,
            max_input_power: Optional[float]=None,
            max_output_power: Optional[float]=None, 
            **kwargs: Mapping[str, Any]
        ):
        super().__init__(efficiency=efficiency, **kwargs)
        self.capacity = capacity
        self.loss_coef = loss_coef
        self.initial_soc = initial_soc
        self.max_input_power = max_input_power
        self.max_output_power = max_output_power

    @property
    def capacity(self) -> float:
        """Thermal Battery maximum capacity [kW]."""
        return self._capacity
    
    @property
    def loss_coef(self) -> float:
        """Thermal Battery capacity loss coefficient between consecutive time steps."""
        return self._loss_coef
    
    @property
    def initial_soc(self) -> float:
        """Battery State of Charge at the beginning of simulation [%]."""
        return self._initial_soc
    
    @property
    def max_input_power(self) -> float:
        """Maximum amount of energy that can be input to the battery at each time step [kWh]."""
        return self._max_input_power
    
    @property
    def max_output_power(self) -> float:
        """Maximum amount of energy that can be output by the battery at each time step [kWh]."""
        return self._max_output_power
    
    @property
    def soc(self) -> np.ndarray:
        """Thermal Battery State of Charge evolution within a simulation episode [%]."""
        return self._soc
    
    @property
    def energy_init(self) -> float:
        """
        Thermal Battery capacity prior to the charge/discharge at each `time_step`[kW].

        NOTE
        ----------
        Takes energy loss proportional to `sefl.loss_coef` into account.
        """
        time_step = max(self.episode_time_step-1, 0)
        return max(0.0, self._soc[time_step]*self._capacity*(1 - self._loss_coef))
    
    @property
    def energy_balance(self) -> np.ndarray:
        """Evolution of the (dis)charged energy by/to the battery within a simulation episode [kW]."""
        return self._energy_balance
        
    @property
    def round_trip_efficiency(self) -> float:
        """Square root of the battery efficiency [kW]."""
        return self._efficiency**0.5
     
    @capacity.setter
    def capacity(self, new_capacity: float):
        assert new_capacity >= 0, f'Invalid capacity {new_capacity}. Must be >= 0.'
        self._capacity = new_capacity

    @loss_coef.setter
    def loss_coef(self, new_coef: float):
        assert 0.0 <= new_coef <= 1.0, f'Invalid capacity {new_coef}. Must be in [0, 1].'
        self._loss_coef = new_coef

    @initial_soc.setter
    def initial_soc(self, new_soc: float):
        assert new_soc is None or 0.0 <= new_soc <= 1.0, f'Invalid capacity {new_soc}. Must be in [0, 1].'
        self._initial_soc = 0.0 if new_soc is None else new_soc

    @max_input_power.setter
    def max_input_power(self, new_power: float):
        assert new_power is None or new_power >= 0, f'Invalid input power {new_power}. Must be >= 0.'
        self._max_input_power = new_power

    @max_output_power.setter
    def max_output_power(self, new_power: float):
        assert new_power is None or new_power >= 0, f'Invalid output power {new_power}. Must be >= 0.'
        self._max_output_power = new_power

    def charge(self, energy: float):
        """
        Charges or discharges storage with respect to specified energy while considering `self.capacity` and `self.soc_init` limitations,
        and energy losses to the environment quantified by `self.round_trip_efficiency`.

        Parameters
        ----------
        :param energy: Energy to charge if (+) or discharge if (-) in [kWh].
        :type energy: float

        NOTE
        ----------
        - If charging, `soc = min(soc_init + energy*round_trip_efficiency, capacity)`
        - If discharging, `soc = max(0, soc_init + energy/round_trip_efficiency)`
        """
        energy_init = self.energy_init
        
        # Compute new energy
        if energy >= 0:
            energy = energy if self.max_input_power is None else np.nanmin([energy, self.max_input_power])
            energy_final = min(energy_init + energy*self.round_trip_efficiency, self.capacity)
        else:
            energy = energy if self.max_output_power is None else np.nanmax([energy, -self.max_output_power]) 
            energy_final = max(0.0, energy_init + energy/self.round_trip_efficiency)

        # Set new SoC
        self._soc[self.episode_time_step] = energy_final/max(self.capacity, 1e-6)

        # Update energy balance
        delta_energy = energy_final - energy_init
        if delta_energy >= 0:
            self._energy_balance[self.episode_time_step] = delta_energy / self.round_trip_efficiency
        else:
            self._energy_balance[self.episode_time_step] = delta_energy * self.round_trip_efficiency

    def reset(self, **kwargs: Mapping[str, Any]):
        """
        Reset the Thermal Battery to its initial state.

        Parameters
        ----------
        :param kwargs: Keyword parameters for `super().reset()`
        :type kwargs: Mapping[str, Any]
        """
        super().reset(**kwargs)
        self._soc = np.zeros(self.episode_length, dtype=np.float32)
        self._soc[0] = self.initial_soc
        self._energy_balance = np.zeros(self.episode_length, dtype=np.float32)

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            **super().get_metadata(),
            'capacity': self.capacity,
            'intial_soc': self.initial_soc,
            'loss_coef': self.loss_coef,
            'max_input_power': self.max_input_power,
            'max_input_power': self.max_output_power,
        }
