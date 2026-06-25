import json
import numpy as np
from collections import defaultdict

from livinglab.envs.livinglab_env import LivingLabEnv
from livinglab.components.device import DualSourceHeatPump, PVSystem
from livinglab.components.battery import ThermalBattery


def init_env_manual() -> LivingLabEnv:
    with open('../config/default.json') as f:
        config = json.load(f)

    # Extract device configurations
    heat_pump_cfgs = config['heat_pump_cfgs']
    thermal_battery_cfgs = config['thermal_battery_cfgs']
    pv_system_cfgs = config['pv_system_cfgs']

    # Initialize devices
    heat_pump = DualSourceHeatPump(**heat_pump_cfgs)
    thermal_battery = ThermalBattery(**thermal_battery_cfgs)
    pv_system = PVSystem(**pv_system_cfgs)

    # Initialize environment
    env = LivingLabEnv(
        seed=config['seed'],
        sim_data_paths=config['sim_data_paths'],
        start_time_step=config['start_time_step'],
        end_time_step=config['end_time_step'],
        heat_pump_cfgs=heat_pump,
        thermal_battery_cfgs=thermal_battery,
        pv_system_cfgs=pv_system,
        dynamics_cfgs=config['dynamics_cfgs'],
        periodic_normalization=config['periodic_normalization'],
        episode_length=24
    )

    return env


def init_env_json() -> LivingLabEnv:
    with open('../config/default.json') as f:
        config = json.load(f)
    
    # Load environment from JSON configuration file and update some configs
    return LivingLabEnv.from_json(config=config, update={'episode_length': 24})


def info():
    env = init_env_json()

    print(f'##### LIVINGLAB ENV #####')
    print(f'#Episodes: {env.simulation_episodes} (length={env.episode_length})\n')
    print(f'Observation names:\n- Active: {env.active_observations}\n- Inactive: {env.inactive_observations}\n')
    print(f'Periodic Observation Metadata:\n{env.periodic_observations_metadata}\n')
    print(f'Observation space:\n{env.observation_space}\n')
    print(f'Action names:\n{env.action_names}\n')
    print(f'Action space:\n{env.action_space}')
    print('##########################')

    _, _ = env.reset()
    obs = env.observations(periodic_normalization=False, names=True)
    print(f'\nObservation at time step {env.time_step}:\n{obs}')

    norm_obs = env.observations(normalize=True, periodic_normalization=True, names=True)
    print(f'\nNormalized Observation at time step {env.time_step}:\n{norm_obs}')

    env.close()


def simple_dual_policy(obs):
    t_air = obs['outdoor_dry_bulb_temperature']
    t_ground = obs['underground_temperature']

    if t_air <= t_ground:
        action = [0.5, 0.0]
    else:
        action = [-0.5, 0.0]

    return action


def simulate(n_episodes: int, source: str):
    assert source.lower() in ['air', 'water', 'dual']

    env = init_env_json()
    kpis_h= defaultdict(list)
    for i in range(n_episodes):
        env.reset()
        while not env.terminated:
            if source == 'Air':
                env.step(actions=[0.5, 0.0])
            elif source == 'Water':
                env.step(actions=[-0.5, 0.0])
            else:
                obs = env.observations(names=True)
                action = simple_dual_policy(obs)
                env.step(actions=action)
            
        kpis_h['reward'].append(env.info['reward']['sum'])
        for k, v in env.info['kpis'].items():
            kpis_h[k].append(v)

    print(f'\n=== RESULTS OVER {n_episodes} SIMULATION EPISODES with {source}-source Heat Pump ===')
    print(f"- Average reward: {np.mean(kpis_h['reward']):.3f}")
    print(f"- Average discomfort: {np.mean(kpis_h['discomfort']):.4f}")
    print(f"- Average electricity consumption: {np.mean(kpis_h['net_electricity_consumption']):.3f}")


    env.close()


if __name__ == '__main__':
    info()    
    simulate(n_episodes=10, source='Air')
    simulate(n_episodes=10, source='Water')
    simulate(n_episodes=10, source='Dual')