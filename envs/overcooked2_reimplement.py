import numpy as np


MAX_NUM_INGREDIENTS = 3
NONE = 0
TOMATO = 1
ONION = 2
DISH = 3
SOUP = 4
ALL_INGREDIENTS = [ONION, TOMATO]

AIR = 0
POT = 1
COUNTER = 2
ONION_SOURCE = 3
DISH_SOURCE = 4
SERVING = 5

TOMATO_SOURCE = 6


def move_in_direction(point, direction, width):
    if direction == Action.NORTH:
        return point - width
    if direction == Action.SOUTH:
        return point + width
    if direction == Action.EAST:
        return point + 1
    if direction == Action.WEST:
        return point - 1
    if direction == Action.STAY:
        return point


class Action(object):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    STAY = 4
    INTERACT = 5
    ALL_ACTIONS = [NORTH, SOUTH, EAST, WEST, STAY, INTERACT]
    NUM_ACTIONS = len(ALL_ACTIONS)


class ObjectState(object):
    def __init__(self, name, num_onions=0, num_tomatoes=0):
        self.name = name
        self.num_onions = num_onions
        self.num_tomatoes = num_tomatoes
        self._cooking_tick = -1

    def num_ingredients(self):
        return self.num_onions + self.num_tomatoes

    def get_recipe(self):
        return (MAX_NUM_INGREDIENTS + 1) * self.num_onions + self.num_tomatoes


class PlayerState(object):
    def __init__(self, position, orientation, held_object=NONE):
        self.position = position
        self.orientation = orientation
        self.proposed_position = position
        self.proposed_orientation = orientation
        self.held_object = held_object

    def has_object(self):
        return self.held_object != NONE

    def get_object(self):
        return self.held_object

    def set_object(self, obj):
        self.held_object = obj

    def remove_object(self):
        obj = self.held_object
        self.held_object = NONE
        return obj

    def update_pos_and_or(self):
        self.position = self.proposed_position
        self.orientation = self.proposed_orientation

    def update_or(self):
        self.orientation = self.proposed_orientation

    def propose_pos_and_or(self, p, o):
        self.proposed_position = p
        self.proposed_orientation = o


class OvercookedState(object):
    def __init__(
        self,
        players,
        objects,
        timestep=0,
    ):
        self.players = tuple(players)
        self.objects = objects
        self.timestep = timestep

    def has_object(self, pos):
        return self.objects[pos] != NONE

    def get_object(self, pos):
        return self.objects[pos]

    def add_object(self, obj, pos):
        self.objects[pos] = obj

    def remove_object(self, pos):
        obj = self.objects[pos]
        self.objects[pos] = NONE
        return obj


class DummyMDP:
    def __init__(
            self,
            terrain,
            height,
            width,
            num_players,
            start_player_x,
            start_player_y,
            placement_in_pot_rew=3,
            dish_pickup_rew=3,
            soup_pickup_rew=5,
            recipe_values=[],
            recipe_times=[],
            horizon=400,
    ):
        self._recipe_values = recipe_values
        self._recipe_times = recipe_times

        self.height = height
        self.width = width
        self.size = height * width
        self.terrain_mtx = terrain
        self.start_player_positions = list(zip(start_player_x, start_player_y))
        self.num_players = num_players

        self.horizon = horizon
        self.placement_in_pot_rew = placement_in_pot_rew
        self.dish_pickup_rew = dish_pickup_rew
        self.soup_pickup_rew = soup_pickup_rew

        self.setup_base_observation()

    def get_terrain(self, pos):
        return self.terrain_mtx[pos]

    def get_time(self, soup):
        return self._recipe_times[soup.get_recipe()]

    def is_cooking(self, soup):
        return soup._cooking_tick >= 0 and soup._cooking_tick < self.get_time(soup)

    def is_ready(self, soup):
        return soup._cooking_tick >= 0 and soup._cooking_tick >= self.get_time(soup)

    def setup_base_observation(self):
        shift = 5 * self.num_players
        self.base_observation = np.zeros([self.size, shift + 10])  # CHANGE: 16 to 10
        for pos in range(self.size):
            v = self.get_terrain(pos)
            if v > AIR:
                self.base_observation[pos, v - 1 + shift] = 1

    def lossless_state_encoding(self, overcooked_state):
        obs = np.copy(self.base_observation)
        shift = 5 * self.num_players

        for pos in range(self.size):
            obj = overcooked_state.get_object(pos)
            if obj == NONE:
                continue

            if obj.name == SOUP:
                if self.get_terrain(pos) == POT:
                    # if obj._cooking_tick < 0:
                    #     obs[pos, shift + 6] = obj.num_onions
                    #     obs[pos, shift + 7] = obj.num_tomatoes
                    # else:
                    #     obs[pos, shift + 8] = obj.num_onions
                    #     obs[pos, shift + 9] = obj.num_tomatoes
                    #     obs[pos, shift + 10] = self.get_time(obj) - obj._cooking_tick
                    #     if self.is_ready(obj):
                    #         obs[pos, shift + 11] = 1
                    obs[pos, shift + 5] = obj.num_onions
                    if obj._cooking_tick < 0:
                        obs[pos, shift + 6] = 0
                    else:
                        obs[pos, shift + 6] = obj._cooking_tick
                else:
                    # obs[pos, shift + 8] = obj.num_onions
                    # obs[pos, shift + 9] = obj.num_tomatoes
                    # obs[pos, shift + 10] = 0
                    # obs[pos, shift + 11] = 1
                    obs[pos, shift + 7] = 1
                # print("SOUP at", pos)
            elif obj.name == DISH:
                # obs[pos, shift + 12] = 1
                obs[pos, shift + 8] = 1
                # print("DISH at", pos)
            elif obj.name == ONION:
                # obs[pos, shift + 13] = 1
                obs[pos, shift + 9] = 1
                # print("ONION at", pos)
            # elif obj.name == TOMATO:
                # obs[pos, shift + 14] = 1

        # if self.horizon - overcooked_state.timestep < 40:
        #     obs[:, shift + 15] = 1

        final_obs_for_players = []

        for primary_agent_idx in range(self.num_players):
            obs_i = np.copy(obs)
            other_i = 1
            for i, player in enumerate(overcooked_state.players):

                pos = player.position

                if i == primary_agent_idx:
                    obs_i[pos, 0] = 1
                    obs_i[pos, self.num_players + player.orientation] = 1
                else:
                    obs_i[pos, other_i] = 1
                    obs_i[pos, self.num_players + 4 * other_i + player.orientation] = 1
                    other_i += 1

                if player.has_object():
                    obj = player.get_object()
                    if obj.name == SOUP:
                        # obs_i[pos, shift + 8] = obj.num_onions
                        # obs_i[pos, shift + 9] = obj.num_tomatoes
                        # obs_i[pos, shift + 10] = 0
                        # obs_i[pos, shift + 11] = 1
                        obs_i[pos, shift + 7] = 1
                        # if primary_agent_idx == 0:
                            # print("h SOUP at", pos)
                    elif obj.name == DISH:
                        obs_i[pos, shift + 8] = 1
                        # if primary_agent_idx == 0:
                            # print("h DISH at", pos)
                    elif obj.name == ONION:
                        obs_i[pos, shift + 9] = 1
                        # if primary_agent_idx == 0:
                            # print("h ONION at", pos)
                    # elif obj.name == TOMATO:
                    #     obs_i[pos, shift + 14] = 1

            final_obs_for_players.append(obs_i)
        # print('DONE')
        return tuple(final_obs_for_players)

    def is_dish_pickup_useful(self, state, non_empty_pots):
        if self.num_players != 2:
            return False

        num_player_dishes = len([p.get_object() for p in state.players if p.has_object() and p.get_object().name == DISH])
        for pos in range(self.size):
            obj = state.get_object(pos)
            if obj != NONE and self.get_terrain(pos) == COUNTER and obj.name == DISH:
                return False
        return num_player_dishes < non_empty_pots

    def get_pot_states(self, state):
        non_empty_pots = 0
        for pos in range(self.size):
            if self.get_terrain(pos) == POT:
                if state.has_object(pos):
                    soup = state.get_object(pos)
                    if soup._cooking_tick >= 0 or soup.num_ingredients() < MAX_NUM_INGREDIENTS:  # Bug in original code?
                        non_empty_pots += 1

        return non_empty_pots

    def deliver_soup(self, state, player, soup):
        player.remove_object()
        return self._recipe_values[soup.get_recipe()]

    def soup_to_be_cooked_at_location(self, state, pos):
        if not state.has_object(pos):
            return False
        obj = state.get_object(pos)
        return (
            obj.name == SOUP
            and not self.is_cooking(obj)
            and not self.is_ready(obj)
            and obj.num_ingredients() > 0
        )

    def soup_ready_at_location(self, state, pos):
        return state.has_object(pos) and self.is_ready(state.get_object(pos))

    def resolve_interacts(self, new_state, joint_action):
        pot_states = self.get_pot_states(new_state)
        reward = [0] * self.num_players

        for player_idx, (player, action) in enumerate(zip(new_state.players, joint_action)):
            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = move_in_direction(pos, o, self.width)
            terrain_type = self.get_terrain(i_pos)

            if terrain_type == COUNTER:
                if player.has_object() and not new_state.has_object(i_pos):
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)
                elif not player.has_object() and new_state.has_object(i_pos):
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)
            elif terrain_type == ONION_SOURCE and player.held_object == NONE:
                player.set_object(ObjectState(ONION))
            elif terrain_type == TOMATO_SOURCE and player.held_object == NONE:
                player.set_object(ObjectState(TOMATO))
            elif terrain_type == DISH_SOURCE and player.held_object == NONE:
                if self.is_dish_pickup_useful(new_state, pot_states):
                    reward[player_idx] += self.dish_pickup_rew
                player.set_object(ObjectState(DISH))
            # elif terrain_type == POT and not player.has_object():
            #     if self.soup_to_be_cooked_at_location(new_state, i_pos):
            #         new_state.get_object(i_pos)._cooking_tick = 0
            elif terrain_type == POT and player.has_object():
                if player.get_object().name == DISH and self.soup_ready_at_location(new_state, i_pos):
                    player.remove_object()
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)
                    reward[player_idx] += self.soup_pickup_rew
                elif player.get_object().name in ALL_INGREDIENTS:
                    if not new_state.has_object(i_pos):
                        new_state.add_object(ObjectState(SOUP, 0, 0), i_pos)
                    soup = new_state.get_object(i_pos)
                    if not (soup._cooking_tick >= 0 or soup.num_onions + soup.num_tomatoes == MAX_NUM_INGREDIENTS):
                        obj = player.remove_object()
                        if obj.name == ONION:
                            soup.num_onions += 1
                        else:
                            soup.num_tomatoes += 1
                        reward[player_idx] += self.placement_in_pot_rew
                    if self.soup_to_be_cooked_at_location(new_state, i_pos) and soup.num_onions + soup.num_tomatoes == MAX_NUM_INGREDIENTS:
                        new_state.get_object(i_pos)._cooking_tick = 0 
            elif terrain_type == SERVING and player.has_object():
                obj = player.get_object()
                if obj.name == SOUP:
                    reward[player_idx] += self.deliver_soup(new_state, player, obj)
        return reward

    def _handle_collisions(self, players):
        for idx0 in range(self.num_players):
            for idx1 in range(idx0 + 1, self.num_players):
                p1_old, p2_old = players[idx0].position, players[idx1].position
                p1_new, p2_new = players[idx0].proposed_position, players[idx1].proposed_position
                if p1_new == p2_new or (p1_new == p2_old and p1_old == p2_new):
                    for p in players:
                        p.update_or()
                    return
        for p in players:
            p.update_pos_and_or()

    def resolve_movement(self, state, joint_action):
        for p, a in zip(state.players, joint_action):
            self._move_if_direction(p, a)
        self._handle_collisions(state.players)

    def step_environment_effects(self, state):
        state.timestep += 1

        for pos in range(self.size):
            obj = state.get_object(pos)
            if obj != NONE and obj.name == SOUP and self.is_cooking(obj):
                obj._cooking_tick += 1

    def get_state_transition(self, state, joint_action):
        rewards = self.resolve_interacts(state, joint_action)
        self.resolve_movement(state, joint_action)
        self.step_environment_effects(state)
        return state, rewards

    def get_standard_start_state(self):
        return OvercookedState(
            [PlayerState(pos[1] * self.width + pos[0], 0) for pos in self.start_player_positions],
            [NONE for _ in range(self.size)]
        )

    def _move_if_direction(self, p, action):
        if action == Action.INTERACT:
            p.propose_pos_and_or(p.position, p.orientation)
        else:
            new_pos = move_in_direction(p.position, action, self.width)
            new_orientation = p.orientation if action == Action.STAY else action
            p.propose_pos_and_or(p.position if self.get_terrain(new_pos) != AIR else new_pos, new_orientation)
