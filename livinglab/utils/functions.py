import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from ladybug.epw import EPW

from typing import Optional, Union, Tuple, List, Mapping


# ASSUMPTION: non-leap year
DAYS_PER_MONTH = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 30, 31, 30, 31], 
    dtype=np.int32
)


class UtilsFunctions:

    @staticmethod
    def sol_air_temperature(
        t_air: Union[np.ndarray, List[float]], 
        g_tot: Optional[Union[np.ndarray, List[float]]]=None,
        g_direct: Optional[Union[np.ndarray, List[float]]]=None,
        g_diffuse: Optional[Union[np.ndarray, List[float]]]=None,
        alpha: float=0.75, 
        h_e: float=23.0
    ) -> np.ndarray:
        """
        Compute the **temperature of the ground at a surface level** as a function of the 
        _air temperature_ and the _solar irradiance_, following the Sol-Air formula [1]:

                            T_sa = T_air + (alpha*G_tot) / h_e

        Parameters
        ----------
        :param alpha: Surface solar absorptivity [W/m^2] (default=0.6 for bare soil).
        :type alpha: float
        :param h_e: External combined heat transfer coefficient [W/m^2K] (default=23.0 [1]).
        :type h_e: float

        References
        ----------
        [1] ASHRAE Handbook - Fundamentals, Ch. 18 (2021).
        """
        # Convert data to numpy arrays
        t_air = np.asarray(t_air, dtype=np.float32)

        if g_tot is not None:
            g_tot = np.asarray(g_tot, dtype=np.float32)
        else:
            assert g_direct is not None and g_diffuse is not None, 'Unsufficient solar irradiance data.'
            g_direct = np.asarray(g_direct, dtype= np.float32)
            g_diffuse = np.asarray(g_diffuse, dtype= np.float32)
            g_tot = g_direct + g_diffuse

        # Calclculate surface level temperature
        ground_surface_temperature = t_air + (alpha + g_tot) / h_e
        return ground_surface_temperature

    @staticmethod
    def extract_kasuda_parameters(path: str) -> Mapping[str, Union[int, float]]:
        """
        Extract the required parameters for the Kasuda model from an annual dataset.

        Parameters
        ----------
        :param path: Path to the annual dataset (assumed to be in .epw format).
        :type path: str

        Return
        ----------
        :return: a distionary containing the Kasuda parameters.
        :rtype: Mapping[str, Union[int, float]]
        """
        # Load the epw file
        epw_file = EPW(path)

        # Extract required data
        t_air = np.array(epw_file.dry_bulb_temperature, dtype=np.float32)
        g_tot = np.array(epw_file.global_horizontal_radiation, dtype=np.float32)

        # Compute annual Sol-Air temperature and parameters
        t_ground = UtilsFunctions.sol_air_temperature(t_air=t_air, g_tot=g_tot)
        t_mean = t_ground.mean()
        amplitude = (t_ground.max() - t_ground.min()) / 2
        min_day = int((t_ground.argmin() + 1) / 24) # <- ASSUMPTION: hourly observations

        return {'mean': t_mean, 'amplitude': amplitude, 't0': min_day}
    
    @staticmethod
    def render_indoor_state(
        ax: Axes, 
        indoor_dry_bulb_temperature: np.ndarray, 
        indoor_temperature_setpoint: np.ndarray,
        comfort_band: np.ndarray,
        outdoor_dry_bulb_temperature: np.ndarray,
        time_steps: int,
    ):
        """
        Render the current indoor state fo the Living Lab.

        Parameters
        ----------
        :param ax: `matplotlib.Axes`
        :type: Axes
        :param indoor_dry_bulb_temperature: Current history of the indoor dry-bulb temperature
        :type indoor_dry_bulb_temperature: np.ndarray
        :param indoor_temperature_setpoint: User-defined setpoint throughout simulation.
        :type indoor_temperature_setpoint: np.ndarray
        :param comfort_band: Maximum deviation from the setpoint in terms of degrees to define comfort.
        :type comfort_band: np.ndarray
        :param outdoor_dry_bulb_temperature: Current history of the outdoor dry-bulb temperature
        :type outdoor_dry_bulb_temperature: np.ndarray
        :param time_steps: Simulation length
        :type time_steps: int 
        """
        # Setpoint and comfort band
        ax.fill_between(
            range(time_steps),
            indoor_temperature_setpoint + comfort_band,
            indoor_temperature_setpoint - comfort_band,
            color='g',
            alpha=0.15,
            label='Comfort band',
        )

        # Current indoor/outoor dry-bulb temperature
        ax.plot(range(len(indoor_dry_bulb_temperature)), indoor_dry_bulb_temperature, linewidth=2.0, label='Indoor Dry Bulb Temperature', color='orange')
        ax.plot(range(len(outdoor_dry_bulb_temperature)), outdoor_dry_bulb_temperature, label='Outdoor Dry Bulb Temperature', color='xkcd:light purple')

        # Style
        ax.grid('on')
        ax.set_title('Indoor Dry-bulb Temperature Evolution', fontweight='bold')
        ax.set_ylabel('Temperature [°C]')
        ax.legend(loc='upper left')

    @staticmethod
    def render_device_control(ax: Axes, thermal_demand: np.ndarray, energy_from_battery: np.ndarray, time_steps: int):
        """
        Render the current device control.

        Parameters
        ----------
        :param ax: `matplotlib.Axes`
        :type: Axes
        :param thermal_demand: Current thermal demand history due to MSHP control.
        :type thermal_demand: np.ndarray
        :param energy_from_battery: Current thermal battery energy balance evolution.
        :type energy_from_battery: np.ndarray
        :param time_steps: Simulation length
        :type time_steps: int 
        """

        def _align_yaxis(ax1: Axes, ax2: Axes):
            y1_lims = ax1.get_ylim()
            y2_lims = ax2.get_ylim()

            y1_frac = (0 - y1_lims[0]) / (y1_lims[1] - y1_lims[0])
            y2_frac = (0 - y2_lims[0]) / (y2_lims[1] - y2_lims[0])

            if y1_frac != y2_frac:
                span = y2_lims[1] - y2_lims[0]
                new_bottom = y2_lims[0] + (y2_frac - y1_frac) * span
                new_top = new_bottom + span
                ax2.set_ylim(new_bottom, new_top)

        # Thermal demand 
        ax.plot(range(len(thermal_demand)), thermal_demand, color='xkcd:soft blue')
        ax.fill_between(
            range(len(thermal_demand)),
            np.zeros_like(thermal_demand),
            thermal_demand,
            color='xkcd:soft blue',
            alpha=0.2
        )
        ax.set_ylabel('Cooling Demand [kWh]')
        ax.yaxis.label.set_color('xkcd:soft blue')

        # Energy from battery
        ax_twin = ax.twinx()
        ax_twin.bar(range(len(energy_from_battery)), energy_from_battery, color='xkcd:orange')
        ax_twin.set_ylabel('Thermal Battery (Dis)Charge [kWh]')
        ax_twin.yaxis.label.set_color('xkcd:orange')

        # Aligning plots
        _align_yaxis(ax, ax_twin)
        ax.plot(np.zeros(time_steps), color='black', linestyle='--')
        ax.grid('on') 
    

class CostFunctions:
    """
    Cost and flexibility functions to evaluate control performance.
    """
    
    @staticmethod
    def ramping(net_electricity_consumption: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Compute the rolling sum of absolute difference in net electric consumption between consecutive time steps.

        Parameters
        ----------
        :param net_electricity_consumption: Electricity consumption time series.
        :type net_electricity_consumption: Union[np.ndarray, List[float]]

        Returns
        ----------
        :return: Ramping cost time series.
        :rtype: np.ndarray
        """ 

        # Compute ramoing between consecutive time steps
        data = pd.DataFrame({'net_electricity_consumption':np.asarray(net_electricity_consumption, dtype=np.float32)})
        data['ramping'] = data['net_electricity_consumption'] - data['net_electricity_consumption'].shift(1)

        # Avoid down ramping
        data['ramping'] = data['ramping'].clip(lower=0)

        # Compute rolling sum of ramping
        data['ramping'] = data['ramping'].rolling(window=data.shape[0],min_periods=1).sum()        
        return data['ramping'].to_numpy(dtype=np.float32)
    
    @staticmethod
    def peak(net_electricity_consumption: Union[np.ndarray, List[float]], window: int=24) -> np.ndarray:
        """
        Compute the average net electricity consumption peak over a specified `window`.

        Parameters
        ----------
        :param net_electricity_consumption: Electricity consumption time series.
        :type net_electricity_consumption: Union[np.ndarray, List[float]]
        :param window: Period window/time steps to find peaks (default is 24).
        :type window: int
            
        Returns
        -------
        :return: average peak over `window` cost.
        :rtype: np.ndarray
        """

        # Group data by window
        data = pd.DataFrame({'net_electricity_consumption':np.asarray(net_electricity_consumption, dtype=np.float32)})
        data['group'] = (data.index/window).astype(int)
        data = data.groupby(['group'])[['net_electricity_consumption']].max()

        # Compute the rolling mean of peak electricity consumption
        data['net_electricity_consumption'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).mean()        
        return data['net_electricity_consumption'].to_numpy(dtype=np.float32)

    @staticmethod
    def electricity_consumption(net_electricity_consumption: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Compute the rolling sum of electricity consumption.

        Parameters
        ----------
        :param net_electricity_consumption: Electricity consumption time series.
        :type net_electricity_consumption: Union[np.ndarray, List[float]]
            
        Returns
        ----------
        :return: cumulative electricity consumption cost time-series.
        :rtype: np.ndarray

        NOTE
        ----------
        Considers also negative net electricity consumption cases representing the export 
        of electical energy to the grid.
        """

        data = pd.DataFrame({'net_electricity_consumption':np.asarray(net_electricity_consumption, dtype=np.float32)})
        data['electricity_consumption'] = data['net_electricity_consumption'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['electricity_consumption'].to_numpy(dtype=np.float32)
    
    @staticmethod
    def cost(cost: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Copmute the rolling sum of monetary costs/earnings due to importing/exporting electricity consumption.

        Parameters
        ----------
        :param cost: Monetary costs time series.
        :type cost: Union[np.ndarray, List[float]]
            
        Returns
        ----------
        :return: cumulative monetary costs time-series.
        :rtype: np.ndarray

        NOTE
        ----------
        Considers also negative net electricity consumption cases representing the export 
        of electical energy to the grid.
        """

        data = pd.DataFrame({'cost':np.asarray(cost, dtype=np.float32)})
        data['cost'] = data['cost'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['cost'].to_numpy(dtype=np.float32)
    
    @staticmethod
    def carbon_emissions(carbon_emissions: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Copmute the rolling sum of carbon emissions due to electricity import from the grid.

        Parameters
        ----------
        :param carbon_emissions: Carbon emissions time series.
        :type carbon_emissions: Union[np.ndarray, List[float]]
            
        Returns
        ----------
        :return: cumulative carbon emissions cost time-series.
        :rtype: np.ndarray
        """

        # Clip only positive values
        data = pd.DataFrame({'carbon_emissions':np.asarray(carbon_emissions, dtype=np.float32).clip(min=0)})
        data['carbon_emissions'] = data['carbon_emissions'].rolling(window=data.shape[0],min_periods=1).sum()
        
        return data['carbon_emissions'].to_numpy(dtype=np.float32)
    
    @staticmethod
    def discomfort(indoor_dry_bulb_temperature: Union[np.ndarray, List[float]], indoor_dry_bulb_setpoint: Union[np.ndarray, List[float]], occupant_count: Union[np.ndarray, List[float]], comfort_band: Union[float, np.ndarray, List[float]]) -> Tuple[np.ndarray]:        
        """
        Compute the rolling discomfort percentage and minimum, maximum and average temperature delta.

        Parameters
        ----------
        :param indoor_dry_bulb_temperature: Indoor temperature time-series.
        :type indoor_dry_bulb_temperature: Union[np.ndarray, List[float]]
        :param indoor_dry_bulb_setpoint: Indoor temperature set point time-series.
        :type indoor_dry_bulb_setpoint: Union[np.ndarray, List[float]]
        :param occupant_count: Occupant count time-series.
        :type occupant_count: Union[np.ndarray, List[float]]
        :param comfort_band: Maximum difference between indoor temperature and setpoint to define comfort.
        :type comfort_band: Union[float, np.ndarray, List[float]]
        """

        # Prepare data
        data = pd.DataFrame({
            'indoor_dry_bulb_temperature': np.asarray(indoor_dry_bulb_temperature, dtype=np.float32),
            'indoor_dry_bulb_setpoint': np.asarray(indoor_dry_bulb_setpoint, dtype=np.float32),
            'occupant_count': np.asarray(occupant_count, dtype=np.float32),
            'comfort_band': comfort_band
        })

        # Compute indoor temperature delta
        data['temperature_delta'] = (data['indoor_dry_bulb_temperature'] - data['indoor_dry_bulb_setpoint']).abs()

        # Do not consider empty bulding scenarios
        occupation_time_steps = data[data['occupant_count'] > 0.0].shape[0]
        data.loc[data['occupant_count'] == 0.0, 'temperature_delta'] = 0.0

        # Compute discomfort over occupied scenarios
        data['discomfort'] = 0
        data.loc[data['temperature_delta'] > data['comfort_band'], 'discomfort'] = 1
        data['discomfort'] = data['discomfort'].rolling(window=data.shape[0],min_periods=1).sum()/occupation_time_steps

        # Compute minimum, maximum and average rolling temperature delta
        data['min_temperature_delta'] = data['temperature_delta'].rolling(window=data.shape[0], min_periods=1).min()
        data['max_temperature_delta'] = data['temperature_delta'].rolling(window=data.shape[0], min_periods=1).max()
        data['avg_temperature_delta'] = data['temperature_delta'].rolling(window=data.shape[0], min_periods=1).mean()

        return (
            data['discomfort'].to_numpy(dtype=np.float32),
            data['min_temperature_delta'].to_numpy(dtype=np.float32),
            data['max_temperature_delta'].to_numpy(dtype=np.float32),
            data['avg_temperature_delta'].to_numpy(dtype=np.float32),
        )