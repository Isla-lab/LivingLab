import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from livinglab.envs import LivingLabEnv


def init_env() -> LivingLabEnv:
    with open('../config/citylearn_challenge_2023_phase_1.json') as f:
        config = json.load(f)
        
    return LivingLabEnv.from_json(config=config)


def info():
    env = init_env()

    print(f'##### LIVINGLAB ENV #####')
    print(f'#Episodes: {env.simulation_episodes} (length={env.episode_length})\n')
    print(f'Observation names:\n{env.observation_names}\n')
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
    env = init_env()

    n_episodes = 100
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