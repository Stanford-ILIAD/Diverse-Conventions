from gym.envs.registration import register

register(
    id='Tree-v0',
    entry_point='tree_env.tree_env:PantheonTree'
)
