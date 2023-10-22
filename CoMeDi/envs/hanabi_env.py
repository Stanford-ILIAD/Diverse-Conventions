from gym.spaces import Discrete, MultiBinary

from pantheonrl_extension.multiagentenv import MultiAgentEnv
from hanabi_learning_environment.rl_env import HanabiEnv
from pantheonrl_extension.vectorenv import MadronaEnv

import build.madrona_python as madrona_python
import build.madrona_hanabi_example_python as hanabi_python

import numpy as np

import torch


DEFAULT_N = 2

FULL_CONFIG = {
            "colors":
                5,
            "ranks":
                5,
            "players":
                DEFAULT_N,
            "max_information_tokens":
                8,
            "max_life_tokens":
                3,
            "observation_type": 1
        }

SMALL_CONFIG = {
            "colors":
                2,
            "ranks":
                5,
            "players":
                DEFAULT_N,
            "hand_size":
                2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type": 1
        }

VERY_SMALL_CONFIG = {
            "colors":
                1,
            "ranks":
                5,
            "players":
                DEFAULT_N,
            "hand_size":
                5, # 2,
            "max_information_tokens":
                3,
            "max_life_tokens":
                1,
            "observation_type": 1
        }


DEFAULT_CONFIG = VERY_SMALL_CONFIG

config_choice = {
    'very_small': VERY_SMALL_CONFIG,
    'small': SMALL_CONFIG,
    'full': FULL_CONFIG
}


class HanabiMadrona(MadronaEnv):

    def __init__(self, num_envs, gpu_id, debug_compile=True, config=None, use_cpu=False, use_env_cpu=False):
        self.config = (config if config is not None else DEFAULT_CONFIG)
        self.hanabi_env = HanabiEnv(config=self.config)
        observation_shape = self.hanabi_env.vectorized_observation_shape()

        # sim = None
        sim = hanabi_python.HanabiSimulator(
            exec_mode = hanabi_python.ExecMode.CPU if use_cpu else hanabi_python.ExecMode.CUDA,
            gpu_id = gpu_id,
            num_worlds = num_envs,
            colors = config["colors"],
            ranks = config["ranks"],
            players = config["players"],
            max_information_tokens = config["max_information_tokens"],
            max_life_tokens = config["max_life_tokens"],
            debug_compile = debug_compile,
        )

        obs_size = observation_shape[0]
        state_size = observation_shape[0] + config['ranks'] * config['colors'] * 5

        self.observation_space = MultiBinary(obs_size)
        self.action_space = Discrete(self.hanabi_env.game.max_moves())

        self.share_observation_space = MultiBinary(state_size)

        device = None
        if use_env_cpu:
            device = torch.device('cpu')
        
        super().__init__(num_envs=num_envs, gpu_id=gpu_id, sim=sim, debug_compile=debug_compile, obs_size=obs_size, state_size=state_size, discrete_action_size=self.hanabi_env.game.max_moves(), env_device=device)
        
        

class PantheonHanabi(MultiAgentEnv):

    def __init__(self, config=None):
        self.config = (config if config is not None else DEFAULT_CONFIG)
        self.hanabi_env = HanabiEnv(config=self.config)

        super().__init__(ego_ind=0, n_players=self.hanabi_env.players)

        observation_shape = self.hanabi_env.vectorized_observation_shape()
        self.observation_space = MultiBinary(observation_shape[0])
        self.action_space = Discrete(self.hanabi_env.game.max_moves())

        self.bits_to_copy = config['ranks'] * config['colors'] * 5
        state_size = observation_shape[0] + self.bits_to_copy

        self.share_observation_space = MultiBinary(state_size) # observation_shape[0] * 2 + 1)  # TODO
    # def get_mask(self):
    #     legal_moves = self.hanabi_env.state.legal_moves()
    #     mask = [False] * self.hanabi_env.game.max_moves()
    #     for m in legal_moves:
    #         mask[self.hanabi_env.game.get_move_uid(m)] = True
    #     return np.array(mask, dtype=bool)

    def get_full_obs(self, obs, player):
        other_obs = np.array(obs['player_observations'][not player]['vectorized'], dtype=bool)
        my_obs = np.array(obs['player_observations'][player]['vectorized'], dtype=bool)
        # player_arr = np.array([player], dtype=bool)
        # share_obs = np.concatenate((my_obs, share_obs, player_arr))
        share_obs = np.concatenate((my_obs, other_obs[:self.bits_to_copy]))
        
        mask = np.zeros(self.hanabi_env.game.max_moves(), dtype=bool)
        mask[obs['player_observations'][player]['legal_moves_as_int']] = True
        return my_obs, share_obs, mask

    def n_step(self, actions):
        move = self.hanabi_env.game.get_move(actions[0]).to_dict()

        obs, reward, done, info = self.hanabi_env.step(move)

        player = obs['current_player']
        return (player,), (self.get_full_obs(obs, player),), tuple([reward] * self.n_players), done, info

    def n_reset(self):
        obs = self.hanabi_env.reset()

        player = obs['current_player']
        return (player,), (self.get_full_obs(obs, player),)


class HanabiState():

    def __init__(self, config):
        self.config = config

    def set(self, state, curagent, verbose):
        config = self.config
        colors = config['colors']
        ranks = config['ranks']
        players = config['players']
        max_information_tokens = config['max_information_tokens']
        max_life_tokens = config['max_life_tokens']
        hand_size = 5
        bitspercard = colors * ranks
        cards_per_color = 4 + (ranks - 2) * 2
        deck_size = cards_per_color * colors

        self.curagent = curagent

        state_size = state.shape[0]

        offset = 0
        # encode hands (including own hand)

        self.hand_sizes = [hand_size, hand_size]
        self.hands = np.zeros((players, hand_size), int)

        for c in range(hand_size):
            sumcards = torch.sum(state[offset:offset+bitspercard])
            if sumcards > 1:
                if verbose:
                    print("Card should only have one value in hand!")
                return False
            elif sumcards == 1 and c >= self.hand_sizes[1-curagent]:
                if verbose:
                    print("Should not have zero card before final card")
                return False
            elif sumcards == 1:
                self.hands[1-curagent, c] = torch.max(state[offset:offset+bitspercard], dim=0).indices.item()
            elif c < self.hand_sizes[1-curagent]:
                self.hand_sizes[1-curagent] = c
            
            offset += bitspercard

        altoffset = state_size-bitspercard*hand_size
        for c in range(hand_size):
            sumcards = torch.sum(state[altoffset:altoffset+bitspercard])
            if sumcards > 1:
                if verbose:
                    print("Card should only have one value in hand!")
                return False
            elif sumcards == 1 and c >= self.hand_sizes[curagent]:
                if verbose:
                    print("Should not have zero card before final card")
                return False
            elif sumcards == 1:
                self.hands[curagent, c] = torch.max(state[altoffset:altoffset+bitspercard], dim=0).indices.item()
            elif c < self.hand_sizes[curagent]:
                self.hand_sizes[curagent] = c
            
            altoffset += bitspercard

        if (state[offset] != (self.hand_sizes[curagent] != hand_size)) or (state[offset + 1] != (self.hand_sizes[1-curagent] != hand_size)):
            if verbose:
                print("Not indicating unfinished hand correctly")
            return False
        offset += 2 # currently, colors * ranks * 5 + players
        
        # encode board

        self.cards_remaining = torch.sum(state[offset:offset+deck_size-hand_size*players], dtype=int)
        if (not torch.all(state[offset:offset+self.cards_remaining])) or (torch.any(state[offset+self.cards_remaining:offset+deck_size-hand_size*players])):
            if verbose:
                print("Board not represented properly")
            return False
        offset += deck_size-hand_size*players

        self.fireworks = np.zeros((colors), int)
        for c in range(colors):
            sumcards = torch.sum(state[offset:offset+ranks], dtype=int)
            self.fireworks[c] = 0 if sumcards == 0 else 1 + torch.max(state[offset:offset+ranks], dim=0).indices.item()
            if sumcards > 1:
                if verbose:
                    print(state[offset:offset+ranks])
                    print(self.fireworks[c])
                    print("Fireworks not represented properly")
                return False
            offset += ranks

        self.information_tokens = torch.sum(state[offset:offset+max_information_tokens], dtype=int)
        if (not torch.all(state[offset:offset+self.information_tokens])) or (torch.any(state[offset+self.information_tokens:offset+max_information_tokens])):
            if verbose:
                print("Info tokens not represented properly")
            return False
        offset += max_information_tokens

        self.life_tokens = torch.sum(state[offset:offset+max_life_tokens], dtype=int)
        if (not torch.all(state[offset:offset+self.life_tokens])) or (torch.any(state[offset+self.life_tokens:offset+max_life_tokens])):
            if verbose:
                print("Life tokens not represented properly")
            return False
        offset += max_life_tokens
        
        # encode discards

        self.discard_counts = np.zeros((colors*ranks), int)
        hand_counts = np.zeros((colors*ranks), int)
        for i in range(2):
            for h in range(self.hand_sizes[i]):
                hand_counts[self.hands[i, h]] += 1
        id = 0
        for c in range(colors):
            for r in range(ranks):
                cr_num = (3 if r == 0 else 1 if r == ranks - 1 else 2)
                self.discard_counts[id] = torch.sum(state[offset:offset+cr_num])
                if (not torch.all(state[offset:offset+self.discard_counts[id]])) or (torch.any(state[offset+self.discard_counts[id]:offset+cr_num])):
                    if verbose:
                        print("Discard cards not represented properly")
                    return False
                if self.discard_counts[id] + hand_counts[id] + (1 if self.fireworks[c] > r else 0) > cr_num:
                    if verbose:
                        print("Too many cards of type:", id)
                        print(self.discard_counts)
                        print(hand_counts)
                        print(self.fireworks)
                        print(self.cards_remaining)
                    return False
                offset += cr_num
                id += 1

        if np.sum(self.discard_counts + hand_counts) + np.sum(self.fireworks) + self.cards_remaining != deck_size:
            if verbose:
                print("Incorrect number of cards; expected", deck_size, "got", np.sum(self.discard_counts + hand_counts) + np.sum(self.fireworks) + self.cards_remaining)
                print(np.sum(self.discard_counts))
                print(np.sum(hand_counts))
                print(np.sum(self.fireworks))
                print(self.cards_remaining)
            return False
        
        # encode last action

        return True

    def simulate_step(self, action, verbose):
        config = self.config
        colors = config['colors']
        ranks = config['ranks']
        players = config['players']
        max_information_tokens = config['max_information_tokens']
        max_life_tokens = config['max_life_tokens']
        hand_size = 5
        bitspercard = colors * ranks
        cards_per_color = 4 + (ranks - 2) * 2
        deck_size = cards_per_color * colors

        curagent = self.curagent

        deckempty = (self.cards_remaining == 0)
        reward = 0

        if action < hand_size:
            # discard
            cardval = self.hands[curagent, action]
            self.discard_counts[cardval] += 1
            self.information_tokens += 1

            self.hands[curagent, action] = -1  # wild card
            if self.cards_remaining != 0:
                self.cards_remaining -= 1
            else:
                self.hand_sizes[curagent] -= 1
        elif action < hand_size + hand_size:
            # play

            # action -= hand_size * 2
            
            cardval = self.hands[curagent, action - hand_size * 2]
            cardcol = cardval // ranks
            cardrank = cardval % ranks

            if self.fireworks[cardcol] == cardrank:
                self.fireworks[cardcol] += 1
                if self.fireworks[cardcol] == ranks:
                    self.information_tokens += 1
                reward += 1
            else:
                self.discard_counts[cardval] += 1
                self.life_tokens -= 1
                # print("PLAYING FAILED CARD")
                # reward -= 1

            self.hands[curagent, action - hand_size * 2] = -1  # wild card
            if self.cards_remaining != 0:
                self.cards_remaining -= 1
            else:
                self.hand_sizes[curagent] -= 1
        elif action < hand_size * 2 + (players - 1) * colors:
            self.information_tokens -= 1
        else:
            self.information_tokens -= 1

        done = False
            
        if self.life_tokens < 1:
            # print("NO MORE LIFE")
            done = True
            reward -= np.sum(self.fireworks)

        if np.sum(self.fireworks) == colors * ranks:
            done = True

        # print("New done")
        
        return True, (done, deckempty, reward)

    def validate_action_masks(self, action_mask, verbose):
        config = self.config
        colors = config['colors']
        ranks = config['ranks']
        players = config['players']
        max_information_tokens = config['max_information_tokens']
        max_life_tokens = config['max_life_tokens']
        hand_size = 5
        bitspercard = colors * ranks
        cards_per_color = 4 + (ranks - 2) * 2
        deck_size = cards_per_color * colors

        curagent = self.curagent

        
        # discard

        offset = 0

        for i in range(hand_size):
            if action_mask[offset] != (i < self.hand_sizes[curagent] and self.information_tokens < max_information_tokens):
                if verbose:
                    print(action_mask[offset])
                    print(self.information_tokens < max_information_tokens)
                    print("Incorrect discard action mask")
                return False
            offset += 1
        
        # play

        for i in range(hand_size):
            if action_mask[offset] != (i < self.hand_sizes[curagent]):
                if verbose:
                    print("Incorrect play action mask")
                return False
            offset += 1

        # reveal color

        for c in range(colors):
            hascolor = False
            for id in self.hands[1-curagent]:
                if id // ranks == c:
                    hascolor = True
            if action_mask[offset] != (self.information_tokens > 0 and hascolor):
                if verbose:
                    print("Incorrect reveal color action mask")
                return False
            offset += 1

        # reveal rank

        for c in range(colors):
            hasrank = False
            for id in self.hands[1-curagent]:
                if id % ranks == c:
                    hasrank = True
            if action_mask[offset] != (self.information_tokens > 0 and hasrank):
                if verbose:
                    print("Incorrect reveal rank action mask")
                return False
            offset += 1
        
        return True

    def equivalent(self, otherstate, verbose):
        if np.any(self.hand_sizes != otherstate.hand_sizes):
            if verbose:
                print("Incorrect hand sizes")
            return False

        for i in range(2):
            for v in self.hands[i]:
                if v != -1 and v not in otherstate.hands[i]:
                    if verbose:
                        print("Incorrect hands (Even with random deck)")
                    return False

        if self.cards_remaining != otherstate.cards_remaining:
            if verbose:
                print("Incorrect cards remaining")
            return False

        if np.any(self.fireworks != otherstate.fireworks):
            if verbose:
                print("Incorrect fireworks")
            return False

        if self.information_tokens != otherstate.information_tokens:
            if verbose:
                print("Incorrect information_tokens remaining")
            return False

        if self.life_tokens != otherstate.life_tokens:
            if verbose:
                print("Incorrect life_tokens remaining")
            return False

        if np.any(self.discard_counts != otherstate.discard_counts):
            if verbose:
                print("Incorrect discard_counts")
            return False
        
        return True

    
def validate_step(states, actions, dones, nextstates, rewards, config, verbose=True):
    numenvs = dones.size(0)

    colors = config['colors']
    ranks = config['ranks']
    players = config['players']
    max_information_tokens = config['max_information_tokens']
    max_life_tokens = config['max_life_tokens']

    hand_size = 5

    bitspercard = colors * ranks

    cards_per_color = 4 + (ranks - 2) * 2
    deck_size = cards_per_color * colors

    # states = states.cpu().numpy()
    # actions = actions.cpu().numpy()
    # dones = dones.cpu().numpy()
    # nextstates = nextstates.cpu().numpy()
    # rewards = rewards.cpu().numpy()
    
    retval = True

    for i in range(numenvs):
        # validate only one agent is active
        if states[0].active[i] == states[1].active[i]:
            if verbose:
                print('Only one agent should be active', states[0].active[i], states[1].active[i], i)
            retval = False
            break

        curagent = 0 if states[0].active[i] else 1

        # must switch if not done
        if not dones[i] and (nextstates[curagent].active[i] or not nextstates[1 - curagent].active[i]):
            if verbose:
                print('Did not switch active agent: old:',
                      states[0].active[i], states[1].active[i],
                      'new:', nextstates[0].active[i], nextstates[1].active[i], i)
            retval = False
            break

        newcuragent = 0 if nextstates[0].active[i] else 1


        # state should be equal to obs (until own hand encoded)
        if not torch.all(nextstates[newcuragent].obs[i] == nextstates[newcuragent].state[i,:-hand_size*bitspercard]):
            if verbose:
                print("State prefix does not match obs")
            retval = False
            break

        # print("THINKING ABOUT", i)
        oldstate = HanabiState(config)
        if not oldstate.set(states[curagent].state[i], curagent, verbose):
            retval = False
            break

        if not oldstate.validate_action_masks(states[curagent].action_mask[i], verbose):
            retval = False
            break

        valid, (truedone, deckempty, truereward) = oldstate.simulate_step(actions[curagent,i], verbose)
        if not valid:
            if verbose:
                print("Error in simulating step")
            retval = False
            break

        if (truedone and not dones[i]) or (dones[i] and not (truedone or deckempty)):
            if verbose:
                print("Difference in dones calculation")
            retval = False
            break

        if truereward != rewards[0, i] or truereward != rewards[1, i]:
            if verbose:
                print(truereward, rewards[:, i], actions[curagent,i])
                print("Difference in rewards expected")
            retval = False
            break
        
        # print("THINKING ABOUT next", actions[curagent][i])

        newstate = HanabiState(config)
        if not newstate.set(nextstates[newcuragent].state[i], newcuragent, verbose):
            print("OLD HANDS", oldstate.hands)
            print("New HANDS", newstate.hands)
            print("curagent", curagent)
            retval = False
            break

        if not newstate.validate_action_masks(nextstates[newcuragent].action_mask[i], verbose):
            retval = False
            break
        
        # validate dones applied correctly
        if dones[i]:
            newcuragent = 0 if nextstates[0].active[i] else 1
            # nextstates must be at some init state

            obs = nextstates[newcuragent].obs[i]
            state = nextstates[newcuragent].state[i]
            
            # encodehands
            offset = 0

            for c in range(hand_size):
                if sum(obs[offset:offset+bitspercard]) != 1:
                    if verbose:
                        print("Card should only have one value in hand!")
                    retval = False
                    break
                offset += bitspercard

            if obs[offset] or obs[offset + 1]:
                if verbose:
                    print("At init, both hands should be full!")
                retval = False
                break
            offset += 2 # currently, colors * ranks * 5 + players
            
            # encodeboard

            if not torch.all(obs[offset: offset + deck_size - hand_size * players]):
                if verbose:
                    print("Incorrect number of cards in initial deck")
                retval = False
                break
            offset += deck_size - hand_size * players

            if torch.any(obs[offset:offset + colors * ranks]):
                if verbose:
                    print("No fireworks should be constructed at init")
                retval = False
                break
            offset += colors * ranks

            if not torch.all(obs[offset: offset + max_information_tokens + max_life_tokens]):
                if verbose:
                    print("All info and life tokens should exist at init")
                retval = False
                break
            offset += max_information_tokens + max_life_tokens
            
            # encodediscards
            if torch.any(obs[offset: offset + deck_size]):
                if verbose:
                    print("No cards should be discarded at init")
                retval = False
                break
            offset += deck_size
            
            # encodelastaction
            if torch.any(obs[offset: offset + players + 4 + players + colors + ranks + hand_size + hand_size + colors * ranks + 2]):
                if verbose:
                    print("No previous move at init")
                retval = False
                break
            offset += players + 4 + players + colors + ranks + hand_size + hand_size + colors * ranks + 2
            
            # encodecardknowledge (won't validate)
            offset += players * hand_size * (colors * ranks + colors + ranks)
            
            # encodeownhand
            for c in range(hand_size):
                if sum(state[offset:offset+bitspercard]) != 1:
                    if verbose:
                        print("Card should only have one value in hand (ego)!", state[offset:offset+bitspercard])
                    retval = False
                    break
                offset += bitspercard
            continue

        # not done; ensure coherence with prior state
        if not oldstate.equivalent(newstate, verbose):
            retval = False
            break
    return retval
