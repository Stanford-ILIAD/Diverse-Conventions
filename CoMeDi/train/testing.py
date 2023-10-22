import shutup; shutup.please()

import sys

from collections import Counter

import torch
import numpy as np

from statistics import stdev

from env_utils import generate_env, set_seed

from partner_agents import DecentralizedAgent
from MAPPO.main_player import MainPlayer
from config import get_config

from pathlib import Path

def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in sorted(Counter(x).items())])

def gen_agent(fload, env, args=None):
    run_dir = Path(fload)

    args.model_dir = str(run_dir / 'models')

    config = {
        'all_args': args,
        'envs': env,
        'device': 'cpu',
        'num_agents': 2,
        'run_dir': run_dir
    }
    ego = MainPlayer(config)
    ego.restore()
    return ego

def run_sim(env, ego, alt):
    env.add_partner_agent(alt)
    rewards = []
    states = []
    fails = 0
    aligned = 0
    obs = env.reset()
    done = False
    reward = 0
    for i in range(200):
        action = ego.get_action(obs, False)
        obs, newreward, done, _ = env.step(action)
        reward += newreward / 20
        # print(newreward)
    rewards.extend(reward.cpu().tolist())
    # states.append(env.state[0])
    # print(env.state)
    print(get_histogram(rewards))
    # print(get_histogram(states))
    print(sum(rewards)/len(rewards))
    print("STDEV:", stdev(rewards) / np.sqrt(len(rewards)))


def main(parser):
    args = parser.parse_args()

    args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name
    
    envs = generate_env(args.env_name, args.n_rollout_threads, args.over_layout)
    ego = DecentralizedAgent(gen_agent(args.ego, envs, args), 0)
    alt = DecentralizedAgent(gen_agent(args.partner, envs, args), 1)
    run_sim(envs, ego, alt)

if __name__ == '__main__':
    parser = get_config()
    parser.add_argument('ego',
                        help='Algorithm for the ego agent')
    parser.add_argument('partner',
                        help='Algorithm for the partner agent')
    main(parser)
