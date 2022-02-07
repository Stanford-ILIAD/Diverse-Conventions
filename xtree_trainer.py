from tree_env.tree_env import PantheonTree
from MAPPO.main_player import MainPlayer
from hanabi_agent import CentralizedAgent
from MAPPO.utils.shared_buffer import SharedReplayBuffer
from XMAPPO.x_player import XPlayer

from config import get_config
import os
from pathlib import Path


def generate_gym(args):
    args.hanabi_name = 'Tree'
    return PantheonTree()


def generate_buffer(args, env):
    eplen = args.episode_length
    args.episode_length = eplen
    buff = SharedReplayBuffer(args,
                              2,
                              env.observation_space,
                              env.share_observation_space,
                              env.action_space)
    args.episode_length = eplen
    return buff


def generate_player(args):
    env = generate_gym(args)
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/tree/results/" + (args.loss_type or "standard") + "/" + str(args.seed))

    print(run_dir)
    config = {
        'all_args': args,
        'env': env,
        'device': 'cpu',
        'num_agents': 2,
        'run_dir': run_dir
    }
    ego = MainPlayer(config)
    partner = CentralizedAgent(ego, 1)
    env.add_partner_agent(partner)
    args.seed += 100
    return ego


def main():
    args = get_config().parse_args()
    print(args)
    N = args.pop_size
    players = [generate_player(args) for _ in range(N)]
    xgym = generate_gym(args)
    xp_buffers = [[generate_buffer(args, xgym) for _ in range(N - i - 1)] for i in range(N)]
    pop_runner = XPlayer(args, players, xgym, xp_buffers, args.episode_length, 1)
    pop_runner.run()


if __name__ == '__main__':
    main()
