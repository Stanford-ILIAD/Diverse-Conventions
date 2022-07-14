from gym.spaces import MultiBinary, Discrete

from pantheonrl.common.multiagentenv import MultiAgentEnv

import numpy as np

DEPTH = 7
STATE_SIZE = 2 ** DEPTH
END_STATE = 2 ** (DEPTH - 1)
SEPARATION = END_STATE + END_STATE // 2


def to_mult_bin(a):
    a = np.array([a], dtype=np.uint8)
    return np.unpackbits(a, count=DEPTH, bitorder='little')


class PantheonTree(MultiAgentEnv):

    def __init__(self):
        super().__init__(ego_ind=0, n_players=2)

        self.observation_space = MultiBinary(DEPTH + 1)
        self.action_space = Discrete(2)

        self.share_observation_space = MultiBinary(DEPTH + 1)

        self.current_player = 0
        self.state = 1
        self.po_states = [1, 1]

    def get_mask(self):
        return np.array([1, 1], dtype=bool)

    def get_full_obs(self):
        my_obs = np.append(to_mult_bin(self.po_states[self.current_player]), self.current_player)
        full_obs = np.append(to_mult_bin(self.state), self.current_player)
        return my_obs, full_obs, self.get_mask()

    def n_step(self, actions):
        move = actions[0]
        self.po_states[self.current_player] = self.po_states[self.current_player] * 2 + move
        self.state = self.state * 2 + move

        self.current_player = 1 - self.current_player

        done = (self.state >= END_STATE)
        if done:
            if self.state < SEPARATION:
                reward = 1 if (self.state % 2 == 0) else 0
            else:
                reward = 3 if (self.state == SEPARATION + 1) else 0
        else:
            reward = 0
        return (self.current_player,), (self.get_full_obs(),), tuple([reward] * self.n_players), done, {}

    def n_reset(self):
        self.current_player = 0
        self.state = 1
        self.po_states = [1, 1]
        return (0,), (self.get_full_obs(),)


class DecentralizedTree(PantheonTree):
    def __init__(self):
        super().__init__()

    def get_full_obs(self):
        obs = super().get_full_obs()
        return obs[0], obs[2]
