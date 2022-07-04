from gym.spaces import Discrete, MultiDiscrete

from pantheonrl.common.multiagentenv import MultiAgentEnv

import numpy as np

NUM_SPACES = 10
VALID_MOVES = [-2, -1, 1, 2]
TIME = 3

class PantheonLine(MultiAgentEnv):

    def __init__(self):
        super(PantheonLine, self).__init__(ego_ind=0, n_players=2)

        self.observation_space = MultiDiscrete([NUM_SPACES, NUM_SPACES, TIME])
        self.action_space = Discrete(len(VALID_MOVES))

        self.share_observation_space = self.observation_space

        self.n_reset()

    def get_mask(self):
        return np.array([1] * len(VALID_MOVES), dtype=bool)

    def get_full_obs(self):
        my_obs = np.append(self.state, self.current_time)
        return (my_obs, my_obs, self.get_mask()), \
               (my_obs[[1, 0, 2]], my_obs[[1, 0, 2]], self.get_mask())

    def n_step(self, actions):
        ego_action = VALID_MOVES[actions[0][0]]
        alt_action = VALID_MOVES[actions[1][0]]

        self.state += np.array([ego_action, alt_action])
        self.current_time -= 1

        done = (self.current_time <= 0)
        reward = 0
        if done:
            reward = -abs(self.state[0] - self.state[1])
        for i in range(2):
            if self.state[i] < 0 or self.state[i] >= NUM_SPACES:
                self.state[i] = 0
                done = True
                reward = -NUM_SPACES
        return (0, 1), self.get_full_obs(), (reward, reward), done, {}

    def n_reset(self):
        self.state = np.random.randint(0, 10, 2)
        self.current_time = TIME-1
        return (0, 1), self.get_full_obs()
