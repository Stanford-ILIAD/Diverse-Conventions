from gym.envs.registration import register

register(
    id='Hanabi-v0',
    entry_point='hanabi_env.hanabi_env:PantheonHanabi'
)

register(
    id='MaskedHanabi-v0',
    entry_point='hanabi_env.hanabi_env:MaskedHanabi'
)
