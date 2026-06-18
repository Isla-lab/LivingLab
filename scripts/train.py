# Omnisafe
from ext import omnisafe

# LivingLab
from livinglab.envs.livinglab_env import LivingLabEnv

# Utils
import os
import torch
import argparse
import yaml, json
from copy import deepcopy as copy
from datetime import datetime
from typing import Any, Mapping


def parse_arguments():
    parser = argparse.ArgumentParser(description='Example on training an RL agent on LivingLab')

    # LivingLab configs
    parser.add_argument('--config_path', type=str, default='../config/device_only_obs.json', help="Path to JSON configuration file")

    # RL configs
    parser.add_argument('--algo', type=str, default='PPO', help="RL algorithm")
    parser.add_argument('--seed', type=int, default=1, help="Experiment seed")
    parser.add_argument('--frac', type=float, default=0.8, help="Fraction of training days")
    parser.add_argument('--episodes', type=int, default=400, help="Training episodes")
    parser.add_argument('--episode_length', type=int, nargs='?', help="Length of an episode in time steps")
    parser.add_argument('--reward_fn', type=str, default='ComfortRewardFunction', help="Reward function class name")

    # Safe RL cost function
    parser.add_argument('--cost_fn', type=str, nargs='?', help="Cost function signature name")
    parser.add_argument('--cost_limit', type=float, nargs='?', help="Cost function violation limit")
    parser.add_argument('--scale', type=float, default=1.0, help="Ccost scale")

    # Wandb logging
    parser.add_argument('--wandb', action='store_true', help="Wandb logging flag")
    parser.add_argument('--project', type=str, nargs='?', help="Wandb project")
    parser.add_argument('--entity', type=str, nargs='?', help="Wandb entity")
    parser.add_argument('--tag', type=str, nargs='*', help="Wandb tag")

    return parser.parse_args()


def train_test_split(env_cfgs: Mapping[str, Any], frac: float):
    assert 0 < frac <= 1, f'Invalid fraction {frac}. Must be in (0,1].'

    # Copy base configurations
    train_cfgs, test_cfgs = copy(env_cfgs), copy(env_cfgs)

    # Total simulation days
    time_steps = env_cfgs['end_time_step'] + 1
    total_days = int(time_steps / 24)

    # Train/test split index
    train_days = int(total_days * frac)
    split_idx = train_days * 24

    # Modify train/test configurations
    train_cfgs['end_time_step'] = split_idx - 1
    train_cfgs['episode_length'] = min(train_cfgs['episode_length'], split_idx)
    if frac < 1:
        test_cfgs['start_time_step'] = split_idx
        test_cfgs['episode_length'] = min(test_cfgs['episode_length'], time_steps - split_idx)

    return train_cfgs, test_cfgs 


def train(args, env_cfgs):
    # Custom Omnisafe configurations
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
            'tag': list(args.tag) if args.tag is not None else []
        },
       
        # --- LIVINGLAB KEY ARGUMENTS ---
        'env_cfgs': {
            **env_cfgs,
        }
    }

    if 'Lag' in args.algo:
        custom_cfgs['env_cfgs'].update({
            'cost_fn': {
                'name': args.cost_fn,
                'kwargs': {
                    'scale': args.scale
                }
            }
        })
        custom_cfgs.update({
            'lagrange_cfgs': {
                'cost_limit': int(env_cfgs['episode_length']*args.cost_limit)*args.scale
            }
        })
        
    # Train the agent
    agent = omnisafe.Agent(args.algo, 'LivingLab-v0', custom_cfgs=custom_cfgs)
    agent.learn()

    return agent


if __name__ == '__main__':
    args = parse_arguments()

    # Experiment logging
    exp_name = datetime.now().strftime('%d-%m-%y_%H:%M')
    exp_dir = f'./experiments/{args.algo}_{exp_name}'
    seed_dir = f'{exp_dir}/seed{args.seed}'
    os.makedirs(seed_dir, exist_ok=True)

    # Save configurations
    with open(f'{exp_dir}/config.yaml', 'w') as f:
        yaml.dump(vars(args), f)

    # Retreive the configurations dictionary from the JSON config file
    env_cfgs = LivingLabEnv.from_json(
        config=args.config_path, 
        update={
            'seed': args.seed, 
            'episode_length': args.episode_length,
            'reward_fn': {
                'class': args.reward_fn
            }
        }, 
        init=False
    )
    train_cfgs, test_cfgs = train_test_split(env_cfgs=env_cfgs, frac=args.frac)

    # Save configurations
    cfgs_dir = f'{exp_dir}/env_cfgs'
    os.makedirs(cfgs_dir, exist_ok=True)
    with open(f'{cfgs_dir}/train_cfgs.json', 'w') as f:
        json.dump(train_cfgs, f, indent=4)
    with open(f'{cfgs_dir}/test_cfgs.json', 'w') as f:
        json.dump(test_cfgs, f, indent=4)

    # Train the agent
    agent = train(args, train_cfgs)

    # Save the policy
    actor = torch.jit.script(agent.agent._actor_critic.actor.net.cpu())
    actor.save(f'{seed_dir}/policy_net.pt')
