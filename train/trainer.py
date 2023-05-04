from MAPPO.main_player import MainPlayer

from config import get_config
import os
from pathlib import Path

from env_utils import generate_env

from partner_agents import CentralizedAgent

import torch

args = get_config().parse_args()

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

envs = generate_env(args.env_name, args.n_rollout_threads, args.over_layout, use_env_cpu=(device=='cpu'))

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

run_dir = (
        os.path.dirname(os.path.abspath(__file__))
        + "/"
        + args.hanabi_name
        + "/results/"
        + (args.run_dir)
        + "/"
        + str(args.seed)
    )
os.makedirs(run_dir, exist_ok=True)
with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
    file.write(str(args))
config = {
    'all_args': args,
    'envs': envs,
    'device': device,
    'num_agents': 2,
    'run_dir': Path(run_dir)
}
ego = MainPlayer(config)
partner = CentralizedAgent(ego, 1)
envs.add_partner_agent(partner)
ego.run()
