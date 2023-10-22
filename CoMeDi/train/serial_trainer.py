import os
from config import get_config

from env_utils import generate_env, set_seed

from XD.serial import run_serial
import torch

args = get_config().parse_args()
device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

set_seed(args.seed, args.cuda_deterministic)
print(args)

pop_size = args.pop_size
env_generator = lambda x : generate_env(args.env_name, x, args.over_layout, use_env_cpu=(device=='cpu'))

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

run_dir = (
    os.path.dirname(os.path.abspath(__file__))
    + "/results/"
    + args.hanabi_name
    + "/"
    + (args.run_dir)
    + "/"
    + str(args.seed)
)
os.makedirs(run_dir, exist_ok=True)
with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
    file.write(str(args))

run_serial(pop_size, args, env_generator, run_dir, device, args.n_rollout_threads, restored=args.restored)
