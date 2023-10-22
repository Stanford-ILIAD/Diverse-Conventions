import os
from config import get_config

from env_utils import generate_env, set_seed

from ADAP.simultaneous import run_parallel
import torch

args = get_config().parse_args()
device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

set_seed(args.seed, args.cuda_deterministic)
print(args)

pop_size = args.pop_size
envs = [generate_env(args.env_name, args.n_rollout_threads, args.over_layout, use_env_cpu=(device=='cpu'))
        for _ in range(pop_size)]

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

run_dir = (
    os.path.dirname(os.path.abspath(__file__))
    + "/results/"
    + args.hanabi_name
    + "/baselines/"
    + (args.loss_type or "None")
    + "/"
    + (args.run_dir)
    + "/"
    + str(args.seed)
)
os.makedirs(run_dir, exist_ok=True)
with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
    file.write(str(args))

print("Calling run_parallel")

run_parallel(pop_size, args, envs, run_dir, device, restored=args.restored)
