"""
Modified from https://github.com/Stanford-ILIAD/Conventions-ModularPolicy/blob/
master/my_gym/envs/hanabi_env.py
"""

from gym.spaces import MultiBinary, Discrete

from pantheonrl.common.multiagentenv import MultiAgentEnv
from hanabi_learning_environment.rl_env import HanabiEnv

import numpy as np


class PantheonHanabi(MultiAgentEnv):

    def __init__(self, config):
        self.config = config
        self.hanabi_env = HanabiEnv(config=self.config)

        super(PantheonHanabi, self).__init__(ego_ind=0,
                                             n_players=self.hanabi_env.players)

        observation_shape = self.hanabi_env.vectorized_observation_shape()
        self.observation_space = MultiBinary(observation_shape[0])
        self.action_space = Discrete(self.hanabi_env.game.max_moves())

    def n_step(self, actions):
        move = self.hanabi_env.game.get_move(actions[0])

        legal_moves = self.hanabi_env.state.legal_moves()
        if not any([str(move) == str(m) for m in legal_moves]):
            move = legal_moves[0]
        move = move.to_dict()

        obs, reward, done, info = self.hanabi_env.step(move)

        player = obs['current_player']
        obs = np.array(obs['player_observations'][player]['vectorized'])
        return (player,), (obs,), tuple([reward] * self.n_players), done, info

    def n_reset(self):
        obs = self.hanabi_env.reset()

        player = obs['current_player']
        obs = np.array(obs['player_observations'][player]['vectorized'])
        return (player,), (obs,)


class MaskedHanabi(MultiAgentEnv):

    def __init__(self, config):
        self.config = config
        self.hanabi_env = HanabiEnv(config=self.config)

        super(MaskedHanabi, self).__init__(ego_ind=0,
                                           n_players=self.hanabi_env.players)

        observation_shape = self.hanabi_env.vectorized_observation_shape()
        self.observation_space = MultiBinary(observation_shape[0])
        self.action_space = Discrete(self.hanabi_env.game.max_moves())

        self.share_observation_space = MultiBinary(observation_shape[0] * 2 + 1)  # TODO

    def get_mask(self):
        legal_moves = self.hanabi_env.state.legal_moves()
        mask = [False] * self.hanabi_env.game.max_moves()
        for m in legal_moves:
            mask[self.hanabi_env.game.get_move_uid(m)] = True
        return np.array(mask, dtype=bool)

    def get_full_obs(self, obs, player):
        other_obs = np.array(obs['player_observations'][not player]['vectorized'], dtype=bool)
        my_obs = np.array(obs['player_observations'][player]['vectorized'], dtype=bool)
        player_arr = np.array([player], dtype=bool)
        if player:
            share_obs = np.concatenate((other_obs, my_obs, player_arr))
        else:
            share_obs = np.concatenate((my_obs, other_obs, player_arr))
        return np.array([my_obs]), np.array([share_obs]), np.array([self.get_mask()])

    def n_step(self, actions):
        move = self.hanabi_env.game.get_move(actions[0]).to_dict()

        obs, reward, done, info = self.hanabi_env.step(move)

        player = obs['current_player']
        return (player,), (self.get_full_obs(obs, player),), tuple([reward] * self.n_players), done, info

    def n_reset(self):
        obs = self.hanabi_env.reset()

        player = obs['current_player']
        return (player,), (self.get_full_obs(obs, player),)
