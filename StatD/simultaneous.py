# from .MCPolicy import MCPolicy
# from .xd_player import XDPlayer
# from MAPPO.utils.shared_buffer import SharedReplayBuffer

import torch
import numpy as np
import random
from pathlib import Path

from MAPPO.main_player import MainPlayer
from DMAPPO.poploss import PopulationLoss, ADAPLoss
from partner_agents import CentralizedAgent

from PopMAPPO.pop_player import PopPlayer

def get_loss(args):
    if args.loss_type is None:
        return PopulationLoss()
    elif args.loss_type == 'ADAP':
        return ADAPLoss(args.loss_param)
    else:
        print("Invalid Loss Type; Assuming no loss")
        return PopulationLoss()

def set_rands(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def generate_player(args, env, run_dir, device):
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
    return ego


def run_parallel(N, args, env, base_dir, device, restored=0):
    print(N, "agents total")
    agent_set = []
    for agent_num in range(N):
        set_rands(args.seed + args.seed_skip * agent_num)
        print("Training agent", agent_num)

        run_dir = Path(base_dir + "/convention" + str(agent_num))

        agent_set += [generate_player(args, env, run_dir, device)]
    pop_runner = PopPlayer(args, agent_set, get_loss(args))
    pop_runner.run()
