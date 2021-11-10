import gym
from stable_baselines3 import PPO

import hanabi_env

from pantheonrl.common.agents import OnPolicyAgent

# han_config = {
#     'colors': 5,
#     'ranks': 5,
#     'players': 2,
#     'hand_size': 5,
#     'observation_type': 1,
#     'seed': 0,
#     'random_start_player': True
# }

han_config={
            "colors":
                2,
            "ranks":
                5,
            "players":
                2,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type":1
        }

env = gym.make('Hanabi-v0', config=han_config)

partner = OnPolicyAgent(PPO('MlpPolicy', env))
env.add_partner_agent(partner)

ego = PPO('MlpPolicy', env, verbose=1)
ego.learn(total_timesteps=100000)
