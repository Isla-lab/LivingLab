import numpy as np
from ladybug.epw import EPW

from typing import Optional, Union, List, Mapping


# ASSUMPTION: non-leap year
DAYS_PER_MONTH = np.array(
    [31, 28, 31, 30, 31, 30, 31, 31, 30, 30, 31, 30, 31], 
    dtype=np.int32
)


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
    t_ground = sol_air_temperature(t_air=t_air, g_tot=g_tot)
    t_mean = t_ground.mean()
    amplitude = (t_ground.max() - t_ground.min()) / 2
    min_day = int(t_ground.argmin() / 24) # <- ASSUMPTION: hourly observations

    return {'mean': t_mean, 'amplitude': amplitude, 't0': min_day}

    