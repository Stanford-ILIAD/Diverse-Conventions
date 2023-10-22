from gym.spaces import Discrete, MultiDiscrete
import build.madrona_python as madrona_python
import build.madrona_balance_example_python as balance_python

from pantheonrl_extension.multiagentenv import MultiAgentEnv
from pantheonrl_extension.vectorenv import MadronaEnv
from pantheonrl_extension.vectorobservation import VectorObservation

import numpy as np

import torch

NUM_SPACES = 5
VALID_MOVES = [-2, -1, 1, 2]
BUFFER = 2
TIME = 3

SCALE = 0.2

MAX_STATES = (NUM_SPACES ** 2)

class BalanceMadronaTorch(MadronaEnv):

    def __init__(self, num_envs, gpu_id, debug_compile=True, use_cpu=False, use_env_cpu=False):
        sim = balance_python.BalanceBeamSimulator(
            exec_mode = balance_python.ExecMode.CPU if use_cpu else balance_python.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_envs,
            debug_compile = debug_compile,
        )

        device = None
        if use_env_cpu:
            device = torch.device('cpu')
            
        super().__init__(num_envs, gpu_id, sim, env_device=device)
        
        self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER] * 2 * TIME + [TIME])
        self.action_space = Discrete(len(VALID_MOVES))

        self.share_observation_space = self.observation_space


from gym.vector.vector_env import VectorEnv
from pantheonrl_extension.vectoragent import RandomVectorAgent

class BalanceGym(VectorEnv):

    def __init__(self, num_envs, gpu_id, debug_compile=True):

        observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER] * 2 * TIME + [TIME])
        action_space = Discrete(len(VALID_MOVES))
        super().__init__(num_envs, observation_space, action_space)

        self.sim = BalanceMadronaTorch(num_envs, gpu_id, debug_compile)
        partner = RandomVectorAgent(lambda: torch.randint_like(self.sim.static_actions[0], high=4))

        self.sim.add_partner_agent(partner)
        self.infos = [{}] * self.num_envs
        # sim = balance_python.BalanceBeamSimulator(
        #     gpu_id = gpu_id,
        #     num_worlds = num_envs,
        #     debug_compile = debug_compile,
        # )
        # super().__init__(num_envs, gpu_id, sim)
        
        # self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER] * 2 * TIME + [TIME])
        # self.action_space = Discrete(len(VALID_MOVES))

        # self.share_observation_space = self.observation_space

    def step(self, actions):
        obs, rew, done, info = self.sim.step(actions[:, None])
        return obs.obs.float(), rew, done, info

    def reset(self):
        return self.sim.reset().obs.float()

    def close(self, **kwargs):
        pass

def generate_state(index):
    # index = index // 2 + 1
    x = index % NUM_SPACES
    y = index // NUM_SPACES
    return np.array([x, y], dtype=int)
    # return np.random.randint(0, NUM_SPACES, 2)

def view(state, time):
    return np.append(state[time:], state[:time])

def unview(state, time):
    return np.append(state[-time:], state[:-time])

class PantheonLine(MultiAgentEnv):
    def __init__(self):
        super().__init__(ego_ind=0, n_players=2)

        # self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER, NUM_SPACES + 2 * BUFFER, TIME])
        self.observation_space = MultiDiscrete([NUM_SPACES + 2 * BUFFER] * 2 * TIME + [TIME])
        self.action_space = Discrete(len(VALID_MOVES))

        self.share_observation_space = self.observation_space

        self.state_ind = -1

        # self.n_reset()

    def update_states(self):
        self.ego_state[self.current_time] = self.state[0] + BUFFER
        self.alt_state[self.current_time] = self.state[1] + BUFFER

    def get_mask(self):
        return np.array([1] * len(VALID_MOVES), dtype=bool)

    def get_full_obs(self):
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
        self.state = np.random.randint(0, NUM_SPACES, 2)  # generate_state(self.state_ind)
        self.ego_state = np.zeros(TIME)
        self.alt_state = np.zeros(TIME)
        self.current_time = TIME - 1
        self.update_states()
        return (0, 1), self.get_full_obs()

    def seed(self, val):
        self.rng = np.random.default_rng(seed=val)


STATIC_ENV = PantheonLine()
    
def validate_step(states, actions, dones, nextstates, rewards, verbose=True):
    states = torch.stack([x.obs for x in states])
    nextstates = torch.stack([x.obs for x in nextstates])
    
    STATIC_ENV.n_reset()
    
    numenvs = dones.size(0)

    states = states.cpu().numpy()
    actions = actions.cpu().numpy()
    dones = dones.cpu().numpy()
    nextstates = nextstates.cpu().numpy()
    rewards = rewards.cpu().numpy()
    
    retval = True
    
    for i in range(numenvs):
        STATIC_ENV.state[0] = states[0][i][0] - BUFFER
        STATIC_ENV.state[1] = states[1][i][0] - BUFFER
        STATIC_ENV.current_time = states[0][i][-1]
        STATIC_ENV.ego_state = unview(states[0][i][:TIME], STATIC_ENV.current_time)
        STATIC_ENV.alt_state = unview(states[1][i][:TIME], STATIC_ENV.current_time)
        _, truenext, truerewards, truedone, _ = STATIC_ENV.n_step(actions[:,i])
        truenext = np.array([truenext[0][0], truenext[1][0]])
        truerewards = np.array([truerewards[0], truerewards[1]])
        # if truedone:
        #     print("FINISHED EPISODE")
        if not np.isclose(truerewards, rewards[:, i]).all():
            if verbose:
                print("start state:", states[:, i], i)
                print("action:", actions[:, i])
                print("madrona transition:", nextstates[:, i])
                print("numpy transition:", truenext)
                print(f"Rewards mismatch: numpy={truerewards}, madrona={rewards[:, i]}")
            retval = False
        
        if truedone != dones[i]:
            if verbose:
                print("start state:", states[:, i], i)
                print("action:", actions[:, i])
                print("madrona transition:", nextstates[:, i])
                print("numpy transition:", truenext)
                print(f"DONES mismatch: numpy={truedone}, madrona={dones[i] == 1}")
            retval = False
            # return False
            # pass
        if dones[i]:
            # print("MADRONA DONE", nextstates[i])
            continue

        if not np.all(np.abs(truenext - nextstates[:,i]) == 0):
            if verbose:
                print("start state:", states[:, i], i)
                print("action:", actions[:, i])
                print("madrona transition:", nextstates[:, i])
                print("numpy transition:", truenext)
                print("TRANSITIONS are not equal")
            retval = False
            # return False
            # pass
    # print("All good")
    return retval
