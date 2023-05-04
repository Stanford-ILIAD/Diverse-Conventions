import gym
import torch
import numpy as np
import os

from pantheonrl_extension.multiagentenv import MultiAgentEnv
from pantheonrl_extension.vectorenv import VectorMultiAgentEnv
from pantheonrl_extension.vectorobservation import VectorObservation

from .overcooked2_reimplement import DummyMDP, Action

import build.madrona_python as madrona_python
import build.madrona_simplecooked_example_python as overcooked_python

LAYOUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "layouts")


def load_dict_from_file(filepath):
    with open(filepath, "r") as f:
        return eval(f.read())


def read_layout_dict(layout_name):
    return load_dict_from_file(os.path.join(LAYOUTS_DIR, layout_name + ".layout"))


class OvercookedMadrona(VectorMultiAgentEnv):

    def __init__(self, layout_name, num_envs, gpu_id, debug_compile=True, use_cpu=False, use_env_cpu=False, ego_agent_idx=0, horizon=200, num_players=None):
        self.layout_name = layout_name
        self.base_layout_params = get_base_layout_params(layout_name, horizon, max_num_players=num_players)
        self.width = self.base_layout_params['width']
        self.height = self.base_layout_params['height']
        self.num_players = self.base_layout_params['num_players']
        self.size = self.width * self.height

        self.horizon = horizon

        sim = overcooked_python.SimplecookedSimulator(
            exec_mode = overcooked_python.ExecMode.CPU if use_cpu else overcooked_python.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_envs,
            debug_compile = debug_compile,
            **self.base_layout_params
        )

        sim_device = torch.device('cpu') if use_cpu or not torch.cuda.is_available() else torch.device('cuda')

        full_obs_size = self.width * self.height * (5 * self.num_players + 10)

        self.sim = sim

        self.static_dones = self.sim.done_tensor().to_torch()
        self.static_active_agents = self.sim.active_agent_tensor().to_torch().to(torch.bool)

        self.static_actions = self.sim.action_tensor().to_torch()
        self.static_observations = self.sim.observation_tensor().to_torch()
        self.static_rewards = self.sim.reward_tensor().to_torch()
        self.static_worldID = self.sim.world_id_tensor().to_torch().to(torch.long)
        self.static_agentID = self.sim.agent_id_tensor().to_torch().to(torch.long)
        self.static_locationWorldID = self.sim.location_world_id_tensor().to_torch().to(torch.long)
        self.static_locationID = self.sim.location_id_tensor().to_torch().to(torch.long)

        self.static_action_mask = torch.ones((num_envs, len(Action.ALL_ACTIONS))).to(device=sim_device, dtype=torch.bool)

        self.obs_size = full_obs_size
        self.state_size = full_obs_size
        
        self.static_scattered_active_agents = self.static_active_agents.detach().clone()
        self.static_scattered_rewards = self.static_rewards.detach().clone()

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents

        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        if use_env_cpu:
            env_device = torch.device('cpu')
        else:
            env_device = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        super().__init__(num_envs, device=env_device, n_players=self.num_players)

        self.static_scattered_observations = torch.empty((self.height * self.width * self.num_players, self.num_envs, 5 * self.num_players + 10), dtype=torch.int8, device=sim_device)
        self.static_scattered_observations[self.static_locationID, self.static_locationWorldID, :] = self.static_observations[:, :, :5 * self.num_players + 10]

        self.infos = [{}] * self.num_envs
        
        self.ego_ind = ego_agent_idx

        self.observation_space = self._setup_observation_space()
        self.share_observation_space = self.observation_space
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

        self.n_reset()

    def _setup_observation_space(self):
        obs_shape = np.array([self.width, self.height, 5 * self.num_players + 10])
        return gym.spaces.MultiBinary(obs_shape)

    def to_torch(self, a):
        return a.to(self.device) #.detach().clone()

    def get_obs(self):
        # self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_locationID, self.static_locationWorldID, :] = self.static_observations[:, :, :5 * self.num_players + 10]

        obs0 = self.to_torch(self.static_scattered_observations[:, :, :]).reshape((self.num_players, self.height, self.width, self.num_envs, -1)).transpose(1, 3)

        obs = [VectorObservation(self.to_torch(self.static_scattered_active_agents[i].to(torch.bool)),
                                 obs0[i],
                                 action_mask=self.static_action_mask)
               for i in range(self.n_players)]

        return obs

    def n_step(self, actions):
        actions_device = self.static_agentID.get_device()
        actions = actions.to(actions_device if actions_device != -1 else torch.device('cpu'))
        self.static_actions.copy_(actions[self.static_agentID, self.static_worldID, :])
        # self.static_actions.copy_(actions)
        self.sim.step()
        
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        return self.get_obs(), self.to_torch(self.static_scattered_rewards), self.to_torch(self.static_dones), self.infos

    def n_reset(self):
        return self.get_obs()

    def close(self, **kwargs):
        pass


MAX_NUM_INGREDIENTS = 3

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
}

TOMATO = "tomato"
ONION = "onion"

AIR = ' '
POT = 'P'
COUNTER = 'X'
ONION_SOURCE = 'O'
TOMATO_SOURCE = 'T'
DISH_SOURCE = 'D'
SERVING = 'S'
TERRAIN_TYPES = [AIR, POT, COUNTER, ONION_SOURCE, DISH_SOURCE, SERVING, TOMATO_SOURCE]
PLAYER_NUMS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
               "!", "@", "#", "$", "%", "^", "&", "*", "(", ")",
               "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
               "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]


def get_true_orders(orders):
    ans = [0] * ((MAX_NUM_INGREDIENTS + 1) ** 2)
    for order in orders:
        num_onions = len([ingredient for ingredient in order['ingredients'] if ingredient == ONION])
        num_tomatoes = len([ingredient for ingredient in order['ingredients'] if ingredient == TOMATO])
        # ans.append((num_onions, num_tomatoes))
        ans[((MAX_NUM_INGREDIENTS + 1) * num_onions + num_tomatoes)] = 1
        # ans.append(num_onions)
        # ans.append(num_tomatoes)
    return ans


def get_base_layout_params(layout_name: str, horizon, max_num_players=None):
    if layout_name.endswith(".layout"):
        base_layout_params = load_dict_from_file(layout_name)
    else:
        base_layout_params = read_layout_dict(layout_name)
    grid = base_layout_params['grid']
    del base_layout_params['grid']

    if 'start_order_list' in base_layout_params:
        # base_layout_params['start_all_orders'] = base_layout_params['start_order_list']
        del base_layout_params['start_order_list']

    if 'num_items_for_soup' in base_layout_params:
        del base_layout_params['num_items_for_soup']

    grid = [layout_row.strip() for layout_row in grid.split('\n')]
    layout_grid = [[c for c in row] for row in grid]

    player_positions = [None] * 64
    for y, row in enumerate(layout_grid):
        for x, c in enumerate(row):
            if c in PLAYER_NUMS:
                layout_grid[y][x] = " "
                idx = PLAYER_NUMS.index(c)
                if max_num_players is None or idx < max_num_players:
                    player_positions[idx] = (x, y)
    num_players = len([x for x in player_positions if x is not None])
    player_positions = player_positions[:num_players]

    base_layout_params['height'] = len(layout_grid)
    base_layout_params['width'] = len(layout_grid[0])
    base_layout_params['terrain'] = [TERRAIN_TYPES.index(x) for row in layout_grid for x in row]
    base_layout_params['num_players'] = len(player_positions)
    # base_layout_params['start_player_positions'] = player_positions
    base_layout_params['start_player_x'] = [p[0] for p in player_positions]
    base_layout_params['start_player_y'] = [p[1] for p in player_positions]

    if 'rew_shaping_params' not in base_layout_params or base_layout_params['rew_shaping_params'] is None:
        base_layout_params['rew_shaping_params'] = BASE_REW_SHAPING_PARAMS

    base_layout_params['placement_in_pot_rew'] = base_layout_params['rew_shaping_params']['PLACEMENT_IN_POT_REW']
    base_layout_params['dish_pickup_rew'] = base_layout_params['rew_shaping_params']['DISH_PICKUP_REWARD']
    base_layout_params['soup_pickup_rew'] = base_layout_params['rew_shaping_params']['SOUP_PICKUP_REWARD']

    del base_layout_params['rew_shaping_params']

    if 'start_all_orders' not in base_layout_params or base_layout_params['start_all_orders'] is None:
        base_layout_params['start_all_orders'] = []

    if 'start_bonus_orders' not in base_layout_params or base_layout_params['start_bonus_orders'] is None:
        base_layout_params['start_bonus_orders'] = []

    original_start_all_orders = base_layout_params['start_all_orders']

    base_layout_params['start_all_orders'] = get_true_orders(original_start_all_orders)
    base_layout_params['start_bonus_orders'] = get_true_orders(base_layout_params['start_bonus_orders'])

    if 'order_bonus' not in base_layout_params:
        base_layout_params['order_bonus'] = 2

    times = [20] * ((MAX_NUM_INGREDIENTS + 1) ** 2)

    if 'onion_time' in base_layout_params and 'tomato_time' in base_layout_params:
        times = []
        for o in range(MAX_NUM_INGREDIENTS + 1):
            for t in range(MAX_NUM_INGREDIENTS + 1):
                times.append(o * base_layout_params['onion_time'] + t * base_layout_params['tomato_time'])
        del base_layout_params['onion_time']
        del base_layout_params['tomato_time']

    if 'recipe_times' in base_layout_params:
        for order, time in zip(original_start_all_orders, base_layout_params['recipe_times']):
            num_onions = len([ingredient for ingredient in order['ingredients'] if ingredient == ONION])
            num_tomatoes = len([ingredient for ingredient in order['ingredients'] if ingredient == TOMATO])
            times[((MAX_NUM_INGREDIENTS + 1) * num_onions + num_tomatoes)] = time

    if 'cook_time' in base_layout_params:
        times = [base_layout_params['cook_time']] * ((MAX_NUM_INGREDIENTS + 1) ** 2)
        del base_layout_params['cook_time']

    base_layout_params['recipe_times'] = times

    values = [20] * ((MAX_NUM_INGREDIENTS + 1) ** 2)

    if 'onion_value' in base_layout_params and 'tomato_value' in base_layout_params:
        values = []
        for o in range(MAX_NUM_INGREDIENTS + 1):
            for t in range(MAX_NUM_INGREDIENTS + 1):
                values.append(o * base_layout_params['onion_value'] + t * base_layout_params['tomato_value'])
        del base_layout_params['onion_value']
        del base_layout_params['tomato_value']

    if 'recipe_values' in base_layout_params:
        for order, value in zip(original_start_all_orders, base_layout_params['recipe_values']):
            num_onions = len([ingredient for ingredient in order['ingredients'] if ingredient == ONION])
            num_tomatoes = len([ingredient for ingredient in order['ingredients'] if ingredient == TOMATO])
            values[((MAX_NUM_INGREDIENTS + 1) * num_onions + num_tomatoes)] = value

    if 'delivery_reward' in base_layout_params:
        # print("BRUH")
        values = [base_layout_params['delivery_reward']] * ((MAX_NUM_INGREDIENTS + 1) ** 2)
        # print(values)
        del base_layout_params['delivery_reward']

    # for i in range(len(values)):
    #     if base_layout_params['start_bonus_orders'][i]:
    #         values[i] *= base_layout_params['order_bonus']

    #     if not base_layout_params['start_all_orders'][i]:
    #         values[i] = 0

    del base_layout_params['order_bonus']

    base_layout_params['recipe_values'] = values

    base_layout_params['horizon'] = horizon

    del base_layout_params['start_all_orders']
    del base_layout_params['start_bonus_orders']

    return base_layout_params


class SimplifiedOvercooked(MultiAgentEnv):
    def __init__(self, layout_name, ego_agent_idx=0, horizon=200, num_players=None):
        self.layout_name = layout_name
        self.base_layout_params = get_base_layout_params(layout_name, horizon, max_num_players=num_players)
        self.width = self.base_layout_params['width']
        self.height = self.base_layout_params['height']
        self.size = self.width * self.height

        self.mdp = DummyMDP(**self.base_layout_params)
        self.horizon = horizon

        super().__init__(ego_ind=ego_agent_idx, n_players=self.base_layout_params['num_players'])

        self.featurize_fn = lambda x: self.mdp.lossless_state_encoding(x)

        self.observation_space = self._setup_observation_space()
        self.share_observation_space = self.observation_space
        self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

        self.n_reset()

    def _setup_observation_space(self):
        obs_shape = np.array([self.mdp.width, self.mdp.height, 5 * self.mdp.num_players + 10])
        return gym.spaces.MultiBinary(obs_shape)

    def get_mask(self):
        return np.array([1] * len(Action.ALL_ACTIONS), dtype=bool)

    def get_obs(self, state):
        obs = self.featurize_fn(state)
        obs = [ob0[:self.size, :].reshape((self.height, self.width, -1)).transpose((1, 0, 2)) for ob0 in obs]
        return [(ob0, ob0, self.get_mask()) for ob0 in obs]

    def n_step(self, actions):
        actions = [act.item() for act in actions]

        next_state, mdp_infos = self.mdp.get_state_transition(self.state, actions)

        self.state = next_state

        done = self.state.timestep >= self.horizon

        reward = sum(mdp_infos)
        info = {}

        return tuple(range(self.mdp.num_players)), self.get_obs(next_state), [reward] * self.mdp.num_players, done, info

    def n_reset(self):
        self.state = self.mdp.get_standard_start_state()
        return tuple(range(self.mdp.num_players)), self.get_obs(self.state)


# class PantheonOvercooked(MultiAgentEnv):

#     def __init__(self, layout_name, ego_agent_idx=0, horizon=200):
#         self.layout_name = layout_name

#         self.mdp = OvercookedGridworld.from_layout_name(self.layout_name, rew_shaping_params=BASE_REW_SHAPING_PARAMS)
#         self.base_env = OvercookedEnv(self.mdp, horizon=horizon)
#         super().__init__(ego_ind=ego_agent_idx, n_players=2)

#         self.featurize_fn = lambda x: self.mdp.lossless_state_encoding(x)

#         self.observation_space = self._setup_observation_space()
#         self.share_observation_space = self.observation_space
#         self.action_space = gym.spaces.Discrete(len(Action.ALL_ACTIONS))

#         self.n_reset()

#     def _setup_observation_space(self):
#         dummy_state = self.mdp.get_standard_start_state()
#         obs_shape = self.featurize_fn(dummy_state)[0].shape
#         return gym.spaces.MultiBinary(obs_shape)

#     def get_mask(self):
#         return np.array([1] * len(Action.ALL_ACTIONS), dtype=bool)

#     def get_obs(self, state):
#         ob0, ob1 = self.featurize_fn(state)
#         return (ob0, ob0, self.get_mask()), (ob1, ob1, self.get_mask())

#     def n_step(self, actions):
#         a0 = Action.INDEX_TO_ACTION[actions[0]]
#         a1 = Action.INDEX_TO_ACTION[actions[1]]

#         next_state, reward, done, info = self.base_env.step((a0, a1))
#         reward = reward + info['shaped_r']

#         return (0, 1), self.get_obs(next_state), (reward, reward), done, info

#     def n_reset(self):
#         self.base_env.reset()
#         return (0, 1), self.get_obs(self.base_env.state)


# def init_validation(layout_name, num_envs, num_players):
#     global VALIDATION_ENVS
#     VALIDATION_ENVS = [
#         PantheonOvercooked(layout_name) for _ in range(num_envs)
#     ]
#     return [x.n_reset()[1] for x in VALIDATION_ENVS]


# def validate_step(states, actions, dones, nextstates, rewards, verbose=True):
#     global VALIDATION_ENVS

#     states = np.array([x.obs.cpu().numpy() for x in states])
#     nextstates = np.array([x.obs.cpu().numpy() for x in nextstates])

#     retval = True
#     for i in range(len(VALIDATION_ENVS)):
#         # assume current state is fine: fix todo?

#         _, truenext, truerewards, truedone, _ = VALIDATION_ENVS[i].n_step(actions[:, i])
#         truenext = np.array([tn[0] for tn in truenext])
#         # print(truenext.shape)
#         # truerewards = np.array(, dtype=np.float32)
#         if not np.all(truerewards == rewards[:, i].cpu().numpy()):
#             if verbose:
#                 # print("start state:", states[:, i], i)
#                 print("action:", actions[:, i])
#                 # print("madrona transition:", nextstates[:, i])
#                 # print("numpy transition:", truenext)
#                 print(f"Rewards mismatch: numpy={truerewards}, madrona={rewards[:, i]}")
#             retval = False

#         if truedone != dones[i]:
#             if verbose:
#                 # print("start state:", states[:, i], i)
#                 print("action:", actions[:, i])
#                 # print("madrona transition:", nextstates[:, i])
#                 # print("numpy transition:", truenext)
#                 print(f"DONES mismatch: numpy={truedone}, madrona={dones[i] == 1}")
#             retval = False
#             # return False
#             # pass
#         if dones[i]:
#             _, truenext = VALIDATION_ENVS[i].n_reset()
#             truenext = np.array([tn[0] for tn in truenext])

#         if not np.all(np.abs(truenext - nextstates[:,i]) == 0):
#             if verbose:
#                 # print("start state:", states[:, i], i)
#                 print("action:", actions[:, i])
#                 print(np.abs(truenext - nextstates[:, i]).nonzero())
#                 print("madrona:", nextstates[:, i][np.abs(truenext - nextstates[:, i]).nonzero()])
#                 print("numpy:", truenext[np.abs(truenext - nextstates[:, i]).nonzero()])
#                 # print("madrona transition:", nextstates[:, i])
#                 # print("numpy transition:", truenext)
#                 print("TRANSITIONS are not equal", i)
#                 print("List of objects", VALIDATION_ENVS[i].base_env.state.objects)
#                 print("ALL_OBJECTS", VALIDATION_ENVS[i].base_env.state.all_objects_list)
#                 print("unowed", VALIDATION_ENVS[i].base_env.state.unowned_objects_by_type)
#                 print('player', VALIDATION_ENVS[i].base_env.state.player_objects_by_type)
#                 # for obj in VALIDATION_ENVS[i].base_env.state.all_objects_list:
#                 #     print(obj.name, obj.position)
#                 print(VALIDATION_ENVS[i].base_env.mdp.lossless_state_encoding(VALIDATION_ENVS[i].base_env.state)[0][np.abs(truenext - nextstates[:, i]).nonzero()[1:]])
#             retval = False

#     return retval
