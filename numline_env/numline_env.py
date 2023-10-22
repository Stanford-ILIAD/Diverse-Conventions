from gym.spaces import Discrete, MultiDiscrete

from pantheonrl.common.multiagentenv import MultiAgentEnv

import numpy as np

NUM_SPACES = 5
VALID_MOVES = [-2, -1, 1, 2]
BUFFER = 2
TIME = 3

SCALE = 0.2

MAX_STATES = (NUM_SPACES ** 2)

def generate_state(index):
    # index = index // 2 + 1
    x = index % NUM_SPACES
    y = index // NUM_SPACES
    return np.array([x, y], dtype=int)
    # return np.random.randint(0, NUM_SPACES, 2)

def view(state, time):
    return np.append(state[time:], state[:time])

class PantheonLine(MultiAgentEnv):
    def __init__(self, do_test=False):
        super().__init__(ego_ind=0, n_players=2)

        # self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER, NUM_SPACES + 2 * BUFFER, TIME])
        self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER] * 2 * TIME + [TIME])
        self.action_space = Discrete(len(VALID_MOVES))

        self.share_observation_space = self.observation_space

        self.state_ind = -1

        self.do_test = do_test

        # self.n_reset()

    def update_states(self):
        self.ego_state[self.current_time] = self.state[0] + BUFFER
        self.alt_state[self.current_time] = self.state[1] + BUFFER

    def get_mask(self):
        return np.array([1] * len(VALID_MOVES), dtype=bool)

    def get_full_obs(self):
        # print(self.state)
        # if self.state[0] == 3 and self.state[1] == 1:
        #     print("NO")
        #     my_obs = np.array([0, 0])
        #     ot_obs = np.array([1, 1])
        # elif self.state[0] == 2 and self.state[1] == 1:
        #     print("YES")
        #     my_obs = np.array([2, 2])
        #     ot_obs = np.array([3, 3])
        # else:
        #     print("WIW")
        #     my_obs = np.array([0, 1])
        #     ot_obs = np.array([2, 3])
        # my_obs = np.array([self.ego_state[self.current_time], self.alt_state[self.current_time], self.current_time])
        # ot_obs = np.array([self.alt_state[self.current_time], self.ego_state[self.current_time], self.current_time])
        ego = view(self.ego_state, self.current_time)
        alt = view(self.alt_state, self.current_time)
        my_obs = np.append(ego, np.append(alt, self.current_time))
        ot_obs = np.append(alt, np.append(ego, self.current_time))
        return (my_obs, my_obs, self.get_mask()), (ot_obs, ot_obs, self.get_mask())

    def n_step(self, actions):
        ego_action = VALID_MOVES[actions[0][0]]
        alt_action = VALID_MOVES[actions[1][0]]

        self.state += np.array([ego_action, alt_action])
        self.current_time -= 1
        self.update_states()

        done = (self.current_time == 0)
        reward = 1.0 if self.state[0] == self.state[1] else -abs(self.state[0] - self.state[1]) * SCALE
        # reward = -abs(self.state[0] - self.state[1])
        for i in range(2):
            if self.state[i] < 0 or self.state[i] >= NUM_SPACES:
                # self.state[i] = 0
                done = True
                reward = -NUM_SPACES * (self.current_time + 1) * SCALE
        return (0, 1), self.get_full_obs(), (reward, reward), done, {}

    def n_reset(self):
        # print("reset")
        self.state_ind = (self.state_ind + 1) % MAX_STATES
        if self.do_test:
            self.state = generate_state(self.state_ind)
        else:
            self.state = np.random.randint(0, NUM_SPACES, 2)
        self.ego_state = np.zeros(TIME)
        self.alt_state = np.zeros(TIME)
        self.current_time = TIME - 1
        self.update_states()
        return (0, 1), self.get_full_obs()


class DecentralizedLine(PantheonLine):
    def __init__(self, do_test=True):
        super().__init__(do_test)

    def get_full_obs(self):
        obs = super().get_full_obs()
        my_obs, ot_obs = obs
        return (my_obs[0], my_obs[2]), (ot_obs[0], ot_obs[2])
