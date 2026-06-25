# LivingLab
from livinglab.envs.livinglab_env import LivingLabEnv

# Utils
import os
import wandb
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
from agent import HourRBC, ComfortRBC


READABLE_KPIS = {
    'discomfort': 'Discomfort [%]',
    'min_indoor_dry_bulb_temperature_delta': 'Minimum Temperature Delta [°C]',
    'max_indoor_dry_bulb_temperature_delta': 'Maximum Temperature Delta [°C]',
    'avg_indoor_dry_bulb_temperature_delta': 'Average Temperature Delta [°C]',
    'ramping': 'Ramping [kWh]',
    'avg_daily_peak': 'Average Daily Peak [kWh]',
    'avg_global_peak': 'Average Global Peak [kWh]',
    'net_electricity_consumption': 'Total Electricity Consumption [kWh]',
    'net_electricity_consumption_cost': 'Total Electricity Consumption Cost [$]',
    'net_electricity_consumption_emissions': 'Total Electricity Consumption Emissions [kgCO2]',
}

def align_yaxis(ax1, ax2):
    y1_lims = ax1.get_ylim()
    y2_lims = ax2.get_ylim()

    # position of zero in axes coordinates
    y1_frac = (0 - y1_lims[0]) / (y1_lims[1] - y1_lims[0])
    y2_frac = (0 - y2_lims[0]) / (y2_lims[1] - y2_lims[0])

    # adjust ax2 limits to match ax1 zero position
    if y1_frac != y2_frac:
        span = y2_lims[1] - y2_lims[0]
        new_bottom = y2_lims[0] + (y2_frac - y1_frac) * span
        new_top = new_bottom + span
        ax2.set_ylim(new_bottom, new_top)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Example on training an RL agent on LivingLab')

    # LivingLab configs
    parser.add_argument('--agent', type=str, default='HourRBC', help="RBC agent to evaluete")
    parser.add_argument('--env_cfgs', type=str, default='./config/default.json', help="Path to the configurations of the environment")
    parser.add_argument('--start', type=int, nargs='?', help="Initial simulation time step")
    parser.add_argument('--end', type=int, nargs='?', help="Ending simulation time step")

    # Wandb logging    
    parser.add_argument('--wandb', action='store_true', help="Wandb logging flag")
    parser.add_argument('--project', type=str, nargs='?', help="Wandb project")
    parser.add_argument('--entity', type=str, nargs='?', help="Wandb entity")
    parser.add_argument('--name', type=str, nargs='?', help="Wandb name")

    return parser.parse_args()


def compare_temperature(args, results):
    # Outdoor temperature
    outdoor_temperature = results['env_h']['temperature']['outdoor_dry_bulb_temperature']

    # Indoor controlled temperature
    indoor_temperature = results['env_h']['temperature']['indoor_dry_bulb_temperature']

    # Cooling demand and Battery support
    cooling_demand = results['env_h']['heat_pump']['cooling_demand']
    energy_balance = results['env_h']['thermal_battery']['energy_balance']

    fig = plt.figure(figsize=[20,10])

    # Set point comfort band
    ax = fig.add_subplot(2,1,1)
    ax.fill_between(
        range(results['env_h']['time_steps']),
        results['env_h']['temperature']['indoor_dry_bulb_temperature_set_point'] + results['env_h']['temperature']['comfort_band'],
        results['env_h']['temperature']['indoor_dry_bulb_temperature_set_point'] - results['env_h']['temperature']['comfort_band'],
        color='g',
        alpha=0.1,
        label='Comfort band',
    )

    # Control temperature
    ax.plot(outdoor_temperature, label='Outdoor Dry Bulb Temperature', color='xkcd:light purple')

    # Control temperature
    ax.plot(indoor_temperature, label='Indoor Dry Bulb Temperature', linewidth=2.0, color='orange')
    ax.grid('on')
    ax.set_title('Indoor Dry-bulb Temperature Evolution', fontweight='bold')
    ax.set_ylabel('Temperature [°C]')
    ax.legend()

    # Thermal demand plot
    bx1 = fig.add_subplot(2,1,2)
    bx1.plot(cooling_demand, color='xkcd:soft blue')
    bx1.fill_between(
        range(results['env_h']['time_steps']),
        np.zeros_like(cooling_demand),
        cooling_demand,
        color='xkcd:soft blue',
        alpha=0.2
    )
    bx1.set_ylabel('Cooling Demand [kWh]')
    bx1.yaxis.label.set_color('xkcd:soft blue')    
    bx2 = bx1.twinx()
    bx2.bar(range(results['env_h']['time_steps']), energy_balance, color='xkcd:orange')
    bx2.set_ylabel('Thermal Battery (Dis)Charge [kWh]')
    bx2.yaxis.label.set_color('xkcd:orange')
    align_yaxis(bx1, bx2)
    bx1.plot(np.zeros_like(cooling_demand), color='black', linestyle='--')
    bx1.grid('on')

    name = args.agent if args.name is None else args.name
    env_name = args.env_cfgs.split('/')[-1].split('.')[0]
    os.makedirs(f'./experiments/{name}/{env_name}/figs', exist_ok=True)
    fig.savefig(f'./experiments/{name}/{env_name}/figs/indoor_dry_bulb_temperature.png', format='png')


def compare_hp_usage(args, results):
    # Outdoor temperatures
    outdoor_temperature = results['env_h']['temperature']['outdoor_dry_bulb_temperature']
    underground_temperature = results['env_h']['temperature']['underground_temperature']

    # Heat Pump control
    hp_signal = results['env_h']['heat_pump']['signal']

    fig = plt.figure(figsize=[20,10])

    # Heat Pump signal
    ax = fig.add_subplot(2,1,1)
    ax.plot(hp_signal, linewidth=2.0, color='xkcd:orange')
    ax.fill_between(
        range(results['env_h']['time_steps']),
        np.zeros_like(hp_signal),
        np.ones_like(hp_signal),
        color='xkcd:light purple',
        alpha=0.1,
        label='Air'
    )
    ax.fill_between(
        range(results['env_h']['time_steps']),
        np.zeros_like(hp_signal),
        -np.ones_like(hp_signal),
        color='xkcd:baby blue',
        alpha=0.1,
        label='Water'
    )
    ax.set_ylabel('MSHP control action [%]')
    ax.legend(
        title='Sources',
        title_fontproperties={'weight': 'bold'}
    )
    ax.set_title('Multi-Source Heat Pump Usage', fontweight='bold')
    ax.grid('on')

    # Temps
    bx = fig.add_subplot(2,1,2)
    bx.plot(outdoor_temperature, color='xkcd:light purple', label='Outdoor Dry-Bulb Temperature')
    bx.plot(underground_temperature, color='xkcd:baby blue', label='Underground Temperature')
    bx.set_ylabel('Temperature [°C]')
    bx.legend()
    bx.grid('on')

    name = args.agent if args.name is None else args.name
    env_name = args.env_cfgs.split('/')[-1].split('.')[0]
    os.makedirs(f'./experiments/{name}/{env_name}/figs', exist_ok=True)
    fig.savefig(f'./experiments/{name}/{env_name}/figs/mshp_usage.png', format='png')


def compare_thermal_battery(args, results):
    # Set figure
    fig = plt.figure(figsize=[20,5])

    bx1 = fig.gca()
    bx1.set_title('Thermal Battery Management')
    # Charge rate
    discharge_rl = results['env_h']['thermal_battery']['energy_balance']
    bx1.bar(range(results['env_h']['time_steps']), discharge_rl, color='xkcd:soft blue', ecolor='xkcd:brick red')
    bx1.set_ylabel('Energy Balance [kW/h]')
    bx1.yaxis.label.set_color('xkcd:soft blue')
    # State of charge
    soc_rl = results['env_h']['thermal_battery']['soc']
    bx2 = bx1.twinx()
    bx2.plot(soc_rl, linewidth=2.0, c='xkcd:orange')
    bx2.set_ylabel('SoC [%]')
    bx2.set_ylim(ymin=-0.05, ymax=1.05)
    bx2.yaxis.label.set_color('xkcd:orange')

    name = args.agent if args.name is None else args.name
    env_name = args.env_cfgs.split('/')[-1].split('.')[0]
    os.makedirs(f'./experiments/{name}/{env_name}/figs', exist_ok=True)
    fig.savefig(f'./experiments/{name}/{env_name}/figs/thermal_battery.png', format='png')


def log_kpis(args, results):

    # Avg reward
    rewards = results['Reward']

    # Avg KPIs
    avg_kpis = {}
    for kpi in READABLE_KPIS.keys():
        values = results['kpis'][kpi]
        avg_kpis[f'KPIs/{READABLE_KPIS[kpi]}'] = values

    # Init wandb run
    run = wandb.init(
        entity=args.entity,
        project='LivingLab_RL_eval_v3' if args.project is None else args.project,
        name=args.agent if args.name is None else args.name
    )

    # Log results
    wandb.log({'Metrics/Reward': rewards})
    wandb.log(avg_kpis)

    # Finish run
    run.finish()


def eval(args, env_cfgs):
    # Load the LivingLabEnv given the configurations
    env = LivingLabEnv.from_json(config=env_cfgs)

    # Modify start and end time step if provided
    if args.start is not None:
        env.start_time_step = args.start
    if args.end is not None:
        env.end_time_step = args.end

    # Load the agent
    if args.agent == 'HourRBC':
        agent = HourRBC(env=env)
    else:
        agent = ComfortRBC(env=env)

    # Episodic return
    ep_reward = 0.0

    # Additional histories
    underground_temperature = []
    hp_signal = []

    # Step through the environment
    env.reset()
    while not env.terminated:
        underground_temperature.append(env.unwrapped.observations(normalize=False, names=True)['underground_temperature'])
        obs = env.observations(periodic_normalization=False, include_all=True, names=True)
        action = agent.predict(obs)
        hp_signal.append(action[0])
        _, reward, _, _, info = env.step(action)
        ep_reward += reward

    # Results
    res = {}
    res['kpis'] = info['kpis']
    res['Reward'] = ep_reward
    res['env_h'] = {
        'time_steps': (env.unwrapped.end_time_step - env.unwrapped.start_time_step),
        'temperature': {
            'underground_temperature': np.array(underground_temperature, dtype=np.float32)[:env.episode_time_step],
            'outdoor_dry_bulb_temperature': env.unwrapped.weather.outdoor_dry_bulb_temperature[env.start_time_step:env.end_time_step],
            'indoor_dry_bulb_temperature': env.unwrapped.energy_simulation.indoor_dry_bulb_temperature[env.start_time_step:env.end_time_step],
            'indoor_dry_bulb_temperature_set_point': env.unwrapped.energy_simulation.indoor_dry_bulb_temperature_cooling_set_point[env.start_time_step:env.end_time_step],
            'comfort_band': env.unwrapped.energy_simulation.comfort_band[env.start_time_step:env.end_time_step]
        },
        'heat_pump': {
            'electricity_consumption': env.unwrapped.heat_pump.electricity_consumption[:env.episode_time_step],
            'cooling_demand': env.unwrapped.energy_simulation.cooling_demand[env.start_time_step:env.end_time_step],
            'signal': np.array(hp_signal, dtype=np.float32)[:env.episode_time_step]
        },
        'thermal_battery': {
            'soc': env.unwrapped.thermal_battery.soc[:env.episode_time_step],
            'energy_balance': env.unwrapped.thermal_battery.energy_balance[:env.episode_time_step]
        }
    }


    # Console log
    table = [['Reward', ep_reward]] + [[READABLE_KPIS[k], v] for k, v in res['kpis'].items()]
    print(f"{'*'*30}\n CONTROL RESULTS (agent={args.agent})")
    print(tabulate(table, headers=['KPI', 'Value'], tablefmt='pretty', colalign=('left', 'left')))
    print(f"{'*'*30}")

    return res


if __name__ == '__main__':
    args = parse_arguments()

    # Load the test configurations
    with open(args.env_cfgs, 'r') as f:
        env_cfgs = json.load(f)

    results = eval(args, env_cfgs)

    compare_temperature(args, results)
    compare_hp_usage(args, results)
    if args.wandb:
        log_kpis(args, results)