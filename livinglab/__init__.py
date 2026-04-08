from gymnasium.envs.registration import register

register(
    id='LivingLab-v0',
    entry_point='livinglab.envs:LivingLabEnv'
)