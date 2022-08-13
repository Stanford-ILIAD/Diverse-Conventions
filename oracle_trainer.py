import os
from config import get_config
import torch
import numpy as np
import random

from pathlib import Path

from tree_env.tree_env import PantheonTree
from numline_env.numline_env import PantheonLine
from hanabi_env.hanabi_env import MaskedHanabi
from overcooked_env.overcooked_env import PantheonOvercooked

from oracle.oracle_player import OraclePlayer
from XD.MCPolicy import MCPolicy
from XD.xd_player import XDPlayer

from MAPPO.utils.shared_buffer import SharedReplayBuffer


def set_rands(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_buffer(args, env):
    return SharedReplayBuffer(
        args, 2, env.observation_space, env.share_observation_space, env.action_space
    )

def generate_gym(args):
    """Generate the gym given the command-line arguments."""
    if args.env_name == "Tree":
        args.hanabi_name = "Tree"
        return PantheonTree()
    if args.env_name == "Line":
        args.hanabi_name = "Line"
        return PantheonLine()
    if args.env_name == "Overcooked":
        args.hanabi_name = "Overcooked"
        return PantheonOvercooked("simple")
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
        return MaskedHanabi(han_config)
    return None


def main():
    args = get_config().parse_args()
    print(args)
    pop_size = args.pop_size
    env = generate_gym(args)
    device = "cpu"
    base_dir = (
        os.path.dirname(os.path.abspath(__file__))
        + "/"
        + args.hanabi_name
        + "/results/"
        + (args.run_dir)
        + "/"
        + str(args.seed)
    )
    os.makedirs(base_dir, exist_ok=True)

    agent_set = []
    set_rands(args.seed)
    for agent_num in range(pop_size):
        next_agent = MCPolicy(
            args,
            env.observation_space,
            env.share_observation_space,
            env.action_space,
            agent_num,
            torch.device(device),
        )

        sp_buf = generate_buffer(args, env)
        xp_buf0 = [generate_buffer(args, env) for _ in range(agent_num)]
        xp_buf1 = [generate_buffer(args, env) for _ in range(agent_num)]
        mp_buf = [generate_buffer(args, env) for _ in range(agent_num)]

        run_dir = Path(base_dir + "/convention" + str(agent_num))

        config = {
            "all_args": args,
            "env": env,
            "device": device,
            "num_agents": 2,
            "run_dir": run_dir,
        }

        runner = XDPlayer(
            config,
            next_agent,
            sp_buf,
            xp_buf0,
            xp_buf1,
            mp_buf,
            agent_set,
            args.xp_weight,
            args.mp_weight,
            args.mix_prob,
            args.env_length,
        )
        runner.model_dir = runner.save_dir
        runner.restore()
        
        agent_set.append(next_agent.actor)

    run_dir = Path(base_dir + "/oracle")
    config = {
        "all_args": args,
        "env": env,
        "device": device,
        "num_agents": 2,
        "run_dir": run_dir,
    }
    ego = OraclePlayer(config, agent_set)
    ego.run()


if __name__ == "__main__":
    main()
