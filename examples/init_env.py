import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from livinglab.envs import LivingLabEnv
from livinglab.components import HeatPump, ThermalBattery, PVSystem


def init_env_manual() -> LivingLabEnv:
    with open('../config/default.json') as f:
        config = json.load(f)

    # Extract device configurations
    heat_pump_cfgs = config['heat_pump_cfgs']
    thermal_battery_cfgs = config['thermal_battery_cfgs']
    pv_system_cfgs = config['pv_system_cfgs']

    # Initialize devices
    heat_pump = HeatPump(**heat_pump_cfgs)
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

    actions = [0.0, 0.0]
    _, reward, terminated, _, info = env.step(actions=actions)
    print(f'\nAction {actions} result:\n - {env.reward_fn.__class__.__name__}: {reward}\n - Terminated: {terminated}\n - Info: {info}')

    next_obs = env.observations(periodic_normalization=False, names=True)
    print(f'\nObservation at time step {env.time_step}:\n{next_obs}')

    env.close()


def sanity_check():
    env = init_env_json()

    n_episodes = 1
    for _ in range(n_episodes):
        env.reset()
        while not env.terminated:
            if env.episode_time_step % 2 == 0:
                env.step(actions=[0.5, 1.0])
            else:
                env.step(actions=[0.2, -1.0])

        print(f'\n[EPISODE {env.episode_counter}-({env.episode_start_time_step}:{env.episode_end_time_step})] LivingLabEnv terminated successfully after {env.episode_length} time steps.')
        for k, v in env.info.items():
            print(f'- {k}: {v}')

    env.close()


if __name__ == '__main__':
    info()
    sanity_check()