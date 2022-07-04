from gym.envs.registration import register

register(
    id='NumLine-v0',
    entry_point='numline_env.numline_env:PantheonLine'
)
