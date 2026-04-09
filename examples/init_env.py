import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import livinglab # <- enable env registration
import gymnasium as gym


def init_env():
    env = gym.make(
        'LivingLab-v0',
        seed=42,
        path='../data/datasets/citylearn_challenge_2023_phase_1',
        start_time_step=0,
        end_time_step=720,
        heat_pump_cfgs={
            'nominal_power': 4.12,
            'efficiency': 0.25,
            'mode': 'cooling',
            'target_temperature': 8.0
        },
        thermal_battery_cfgs={
            'capacity': 4.0,
            'efficiency': 0.95,
            'loss_coef': 1e-05
        },
        pv_system_cfgs={
            'nominal_power': 2.4
        },
        dynamics_cfgs={
            'path': '../data/datasets/citylearn_challenge_2023_phase_1/Building_1.pth',
            'num_layers': 2,
            'input_size': 13,
            'hidden_size': 16,
            'lookback': 12,
            'input_observation_names': [
                "direct_solar_irradiance",
                "diffuse_solar_irradiance",
                "outdoor_dry_bulb_temperature",
                "indoor_dry_bulb_temperature_cooling_set_point",
                "occupant_count",
                "cooling_demand",
                "month_sin",
                "month_cos",
                "hour_sin",
                "hour_cos",
                "day_type_sin",
                "day_type_cos",
                "indoor_dry_bulb_temperature"
            ],
            'input_norm_min': [
                0.0,
                0.0,
                21.7,
                18.88889,
                0.0,
                0.0,
                -0.8660254,
                -1.0,
                -1.0,
                -1.0,
                -0.9749279,
                -0.90096885,
                12.83158
            ],
            "input_norm_max": [
                931.0,
                486.5,
                42.8,
                24.444445,
                3.0,
                12.6740625,
                1.2246469000000002e-16,
                -0.5,
                1.0,
                1.0,
                0.9749279,
                1.0,
                43.536755
            ],
        },
        periodic_normalization=False
    )

    print(f'##### LIVINGLAB ENV #####')
    print(f'Observation names:\n{env.unwrapped.observation_names}\n')
    print(f'Observation space:\n{env.observation_space}\n')
    print(f'Action names:\n{env.unwrapped.action_names}\n')
    print(f'Action space:\n{env.action_space}')
    print('##########################')

    obs, _ = env.reset()
    print(f'\nObservation at time step {env.unwrapped.time_step}:\n{obs}')

    env.close()


if __name__ == '__main__':
    init_env()