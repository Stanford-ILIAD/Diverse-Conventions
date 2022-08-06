import sys

from collections import Counter

import torch

from tree_env.tree_env import DecentralizedTree
from tree_env.tree_agent import SafeAgent, RiskyAgent, RandomTreeAgent

from numline_env.numline_env import DecentralizedLine
from numline_env.numline_agent import (RandomLineAgent, LeftBiasAgent, RightBiasAgent)

from overcooked_env.overcooked_env import DecentralizedOvercooked

from partner_agents import DecentralizedAgent
from MAPPO.r_actor_critic import R_Actor
from config import get_config

EGO_LIST = ['RAND', 'LEFT', 'RIGHT', 'SAFE', 'RISKY', 'LOAD']

def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in sorted(Counter(x).items())])

def generate_gym(args):
    """Generate the gym given the command-line arguments."""
    if args.env_name == "Tree":
        args.hanabi_name = "Tree"
        return DecentralizedTree()
    if args.env_name == "Line":
        args.hanabi_name = "Line"
        return DecentralizedLine()
    if args.env_name == "Overcooked":
        args.hanabi_name = "Overcooked"
        return DecentralizedOvercooked("simple")
    if args.env_name == "Hanabi":
        han_config = {
            "colors": args.han_colors,
            "ranks": args.han_ranks,
            "players": 2,
            "hand_size": args.han_hand,
            "max_information_tokens": args.han_info,
            "max_life_tokens": args.han_life,
            "observation_type": 1,
        }
        return None  # TODO
    return None

def gen_agent(value, env, envname, fload, args=None):
    if value == 'RAND':
        return RandomLineAgent() if envname == "Line" else RandomTreeAgent
    if value == 'LEFT':
        return LeftBiasAgent()
    if value == 'RIGHT':
        return RightBiasAgent()
    if value == 'SAFE':
        return SafeAgent()
    if value == 'RISKY':
        return RiskyAgent()
    if value == 'LOAD':
        actor = R_Actor(args, env.observation_space, env.action_space)
        print(fload)
        if fload is None:
            print("NEED TO INPUT FILE")
            sys.exit()
        state_dict = torch.load(fload)
        actor.load_state_dict(state_dict)
        return DecentralizedAgent(actor)

def run_sim(env, ego, alt):
    env.add_partner_agent(alt)
    rewards = []
    for _ in range(100):
        # print(f'Game #{game}')
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward
        rewards.append(reward)
    print(get_histogram(rewards))
    print(sum(rewards)/len(rewards))

def main(parser):
    args = parser.parse_args()
    
    env = generate_gym(args)
    ego = gen_agent(args.ego, env, args.env_name, args.ego_load, args)
    alt = gen_agent(args.partner, env, args.env_name, args.partner_load, args)
    run_sim(env, ego, alt)

if __name__ == '__main__':
    parser = get_config()
    parser.add_argument('ego',
                        choices=EGO_LIST,
                        help='Algorithm for the ego agent')
    parser.add_argument('--ego-load',
                        help='File to load the ego agent from')
    parser.add_argument('partner',
                        choices=EGO_LIST,
                        help='Algorithm for the partner agent')
    parser.add_argument('--partner-load',
                        help='File to load the partner agent from')
    main(parser)
