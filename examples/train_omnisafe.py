# Omnisafe
from ext import omnisafe

# LivingLab
from livinglab.envs.livinglab_env import LivingLabEnv

# Utils
import torch
import argparse


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
    parser.add_argument('--project', type=str, nargs='?', help="Wandb project")
    parser.add_argument('--entity', type=str, nargs='?', help="Wandb entity")

    return parser.parse_args()


def main(args):
    # 1. Retreive the configurations dictionary from the JSON config file
    env_cfgs = LivingLabEnv.from_json(config=args.config_path, update={'seed': args.seed, 'episode_length': args.episode_length}, init=False)

    # 2. Define omnisafe configurations
    custom_cfgs = {
        'seed': args.seed,
        'train_cfgs': {
            'total_steps': args.episodes*(env_cfgs['episode_length']-1),
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        },
        'algo_cfgs': {
            'steps_per_epoch': env_cfgs['episode_length']-1,
            'obs_normalize': False # <- DO NOT SET THIS TO True!
        },
        'model_cfgs': {
            'actor_type': 'gaussian_sac',
        },
        'logger_cfgs': {
            'use_wandb': args.wandb,
            'wandb_project': args.project if args.project is not None else 'None',
            'entity': args.entity if args.entity is not None else 'None',
            'mode': 'online',
        },
       
        # --- LIVINGLAB KEY ARGUMENTS ---
        'env_cfgs': {
            **env_cfgs
        }
    }

    # 3. Define and train the agent
    agent = omnisafe.Agent('PPO', 'LivingLab-v0', custom_cfgs=custom_cfgs)
    agent.learn()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
