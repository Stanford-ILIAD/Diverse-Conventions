from gym.envs.registration import register

register(
    id='PantheonOvercooked-v0',
    entry_point='overcooked_env.overcooked_env:PantheonOvercooked'
)
