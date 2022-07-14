from gym.spaces import Discrete, MultiDiscrete

from pantheonrl.common.multiagentenv import MultiAgentEnv

import numpy as np

NUM_SPACES = 10
VALID_MOVES = [-2, -1, 1, 2]
TIME = 3


class PantheonLine(MultiAgentEnv):
    def __init__(self):
        super().__init__(ego_ind=0, n_players=2)

        self.observation_space = MultiDiscrete([NUM_SPACES, NUM_SPACES] * TIME + [TIME])
        self.action_space = Discrete(len(VALID_MOVES))

        self.share_observation_space = self.observation_space

        self.n_reset()

    def update_states(self):
        self.ego_state[self.current_time] = self.state[0]
        self.alt_state[self.current_time] = self.state[1]

    def get_mask(self):
        return np.array([1] * len(VALID_MOVES), dtype=bool)

    def get_full_obs(self):
        my_obs = np.append(self.ego_state, np.append(self.alt_state, self.current_time))
        ot_obs = np.append(self.alt_state, np.append(self.ego_state, self.current_time))
        return (my_obs, my_obs, self.get_mask()), (ot_obs, ot_obs, self.get_mask())

    def n_step(self, actions):
        ego_action = VALID_MOVES[actions[0][0]]
        alt_action = VALID_MOVES[actions[1][0]]

        self.state += np.array([ego_action, alt_action])
        self.current_time -= 1
        self.update_states()

        done = self.current_time <= 0
        reward = -abs(self.state[0] - self.state[1])
        for i in range(2):
            if self.state[i] < 0 or self.state[i] >= NUM_SPACES:
                # self.state[i] = 0
                done = True
                reward = -NUM_SPACES * (self.current_time + 1)
        return (0, 1), self.get_full_obs(), (reward, reward), done, {}

    def n_reset(self):
        self.state = np.random.randint(0, NUM_SPACES, 2)
        self.ego_state = np.zeros(TIME)
        self.alt_state = np.zeros(TIME)
        self.current_time = TIME - 1
        self.update_states()
        return (0, 1), self.get_full_obs()


class DecentralizedLine(PantheonLine):
    def __init__(self):
        super().__init__()

    def get_full_obs(self):
        obs = super().get_full_obs()
        my_obs, ot_obs = obs
        return (my_obs[0], my_obs[2]), (ot_obs[0], ot_obs[2])
