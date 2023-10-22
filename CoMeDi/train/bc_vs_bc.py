import shutup; shutup.please()

import sys

from collections import Counter

import torch
import numpy as np

from statistics import stdev

from env_utils import generate_env, set_seed

from partner_agents import DecentralizedAgent, TFJSAgent
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

def get_path(layout):
    if layout == 'simple':
        return "torch_results/models/pbt_cramped_room_agent.pt"
    elif layout == 'random1':
        return "torch_results/models/pbt_coordination_ring_agent.pt"
    elif layout == 'random3':
        return "torch_results/models/pbt_counter_circuit_agent.pt"
    elif layout == "unident_s":
        return "torch_results/models/pbt_asymmetric_advantages_agent.pt"
    elif layout == "random0":
        return "torch_results/models/pbt_forced_coordination_agent.pt"

def get_path1(layout):
    if layout == 'simple':
        return "torch_results/models/ppo_bc_cramped_room_agent.pt"
    elif layout == 'random1':
        return "torch_results/models/ppo_bc_coordination_ring_agent.pt"
    elif layout == 'random3':
        return "torch_results/models/ppo_bc_counter_circuit_agent.pt"
    elif layout == "unident_s":
        return "torch_results/models/ppo_bc_asymmetric_advantages_agent.pt"
    elif layout == "random0":
        return "torch_results/models/ppo_bc_forced_coordination_agent.pt"

def run_sim(env, ego, alt):
    env.add_partner_agent(alt)
    rewards = []
    states = []
    fails = 0
    aligned = 0
    obs = env.reset()
    done = False
    reward = 0
    for _ in range(10):
        for i in range(200):
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward / 20
            # print(newreward)
        rewards.extend(reward.cpu().tolist())
        reward = 0
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
    alt = TFJSAgent(get_path(args.over_layout), 1)
    ego = TFJSAgent(get_path(args.over_layout), 1)
    run_sim(envs, ego, alt)

if __name__ == '__main__':
    parser = get_config()
    main(parser)
