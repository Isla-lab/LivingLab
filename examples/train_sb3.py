import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings; warnings.filterwarnings("ignore", category=UserWarning)

# SB3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# LivingLab imports
from livinglab.envs.livinglab_env import LivingLabEnv
from livinglab.utils.wrappers import NormalizedSpaceWrapper

# Utils
import wandb
import argparse
from collections import defaultdict, deque
from tabulate import tabulate
from datetime import datetime


class KPICallback(BaseCallback):
    def __init__(self, verbose: int=0, wandb: bool=False, window_len: int=100):
        super().__init__(verbose)

        # Wandb logging
        self.wandb = wandb
        self.log_fn = lambda x: sum(x)/len(x)

        # Episodic info
        self.ep_count = 0
        self.ep_rewards = deque(maxlen=window_len)
        self.ep_lengths = deque(maxlen=window_len)

        # KPIs info
        self.kpis_h = defaultdict(lambda: deque(maxlen=window_len))
        self.READABLE_KPIS = {
            'discomfort': 'Discomfort [%]',
            'indoor_dry_bulb_temperature_delta': 'Total Temperature Delta [°C]',
            'avg_indoor_dry_bulb_temperature_delta': 'Average Temperature Delta [°C]',
            'net_electricity_consumption': 'Total Electricity Consumption [kWh]',
            'avg_net_electricity_consumption': 'Average Electricity Consumption [kWh]',
            'net_electricity_consumption_cost': 'Total Electricity Consumption Cost [$]',
            'avg_net_electricity_consumption_cost': 'Average Electricity Consumption Cost [$]',
            'net_electricity_consumption_emissions': 'Total Electricity Consumption [kgCO2]',
            'avg_net_electricity_consumption_emissions': 'Average Electricity Consumption [kgCO2]'
        }

    def _on_step(self) -> bool:

        info = self.locals['infos'][0]
        if self.locals['dones'][0]:
            # Retrieve episodic info
            ep_info = info['episode']
            self.ep_rewards.append(ep_info['r'])
            self.ep_lengths.append(ep_info['l'])
            if self.wandb:
                wandb_log = {
                    'TotalEnvSteps': self.num_timesteps,
                    'Training/EpReward': self.log_fn(self.ep_rewards),
                    'Training/EpLength': self.log_fn(self.ep_lengths)
                }

            # Retrieve KPIs
            kpis = info['kpis']
            kpi_table = []
            for k, v in kpis.items():
                key = self.READABLE_KPIS[k]
                kpi_table.append([key, f"{f'{v*100:.2f}' if k == 'discomfort' else v}"])
                self.kpis_h[k].append(v)
                if self.wandb:
                    wandb_log.update({f'KPIs/{key}': self.log_fn(self.kpis_h[k])})

            # Log to wandb
            if self.wandb:
                wandb.log(data=wandb_log, step=self.ep_count)

            # Log to terminal
            print(
                f"{'#'*30}\nEPISODE {self.ep_count}" +
                f"\n - Reward: {ep_info['r']}"       +
                f"\n - Length: {ep_info['l']}"
            )
            print(tabulate(kpi_table, headers=['KPI', 'Value'], tablefmt='pretty', colalign=('left', 'left')))

            # Update episode count
            self.ep_count += 1

        return True


def parse_arguments():
    parser = argparse.ArgumentParser(description='Example on training an RL agent on LivingLab')

    # LivingLab configs
    parser.add_argument('--config_path', type=str, default='../config/default.json', help="Path to JSON configuration file")

    # RL configs
    parser.add_argument('--seed', type=int, default=1, help="Experiment seed")
    parser.add_argument('--episodes', type=int, default=1000, help="Training episodes")
    parser.add_argument('--episode_length', type=int, nargs='?', help="Length of an episode in time steps")

    # Wandb logging
    parser.add_argument('--wandb', action='store_true', help="Wandb logging flag")
    parser.add_argument('--project', type=str, default='LivingLab_RL_training', help="Wandb project")
    parser.add_argument('--entity', type=str, default='universitaverona', help="Wandb entity")

    return parser.parse_args()


def main(args):
    # 1. Initialize and wrap the environment for observation/action normalziation
    env = LivingLabEnv.from_json(config=args.config_path, update={'seed': args.seed, 'episode_length': args.episode_length})
    env = NormalizedSpaceWrapper(env)

    # 2. Initialize the callback and optional wandb run
    callback = KPICallback(wandb=args.wandb)
    if args.wandb:
        run = wandb.init(
            project=args.project,
            entity=args.entity,
            group='sb3',
            name=f"PPO_SB3_seed{args.seed}_{datetime.now().strftime('%d-%m-%y_%H:%M:%S')}"
        )

    # 3. Initialize SB3 agent
    agent = PPO(
        policy='MlpPolicy', 
        env=env,
        n_steps=env.unwrapped.episode_length,
        batch_size=64,      #
        n_epochs=40,        #
        gamma=0.99,         #
        gae_lambda=0.95,    #
        clip_range=0.2,     # -> from omnisafe/omnisafe/configs/on-policy/PPO.yaml
        ent_coef=0.0,       #
        vf_coef=0.001,      #
        max_grad_norm=40.0, #
        target_kl=0.02,     #
        seed=args.seed
    )

    # 4. Train the agent
    _ = agent.learn(
        total_timesteps=env.unwrapped.episode_length*args.episodes,
        callback=callback
    )

    # 5. (Optional) Terminate wandb run
    if args.wandb:
        run.finish()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)