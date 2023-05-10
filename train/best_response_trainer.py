import os
from config import get_config

from env_utils import generate_env, set_seed

import torch

from XD.MCPolicy import MCPolicy
from XD.xd_player import XDPlayer
from MAPPO.utils.shared_buffer import SharedReplayBuffer

# import torch
import numpy as np
import random
from pathlib import Path

def set_rands(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_buffer(args, env, device, num_threads=None):
    return SharedReplayBuffer(
        args, 2, env.observation_space, env.share_observation_space, env.action_space, device, num_threads
    )

def run_serial(N, args, env_generator, base_dir, device, threads, restored=0):
    args.xp_weight = -0.25
    args.mp_weight = 0
    print(N, "agents total")
    agent_set = []
    true_recurrence = args.use_recurrent_policy
    args.use_recurrent_policy = False
    args.use_average = True
    for agent_num in range(N + 1):
        env = env_generator(threads * ((agent_num) * 2 + 1))
        env_mp = env_generator(args.env_length - 1)
        set_rands(args.seed + args.seed_skip * agent_num)
        print("Training agent", agent_num)
        if agent_num == N:
            args.use_recurrent_policy = true_recurrence
        next_agent = MCPolicy(
            args,
            env.observation_space,
            env.share_observation_space,
            env.action_space,
            agent_num,
            torch.device(device),
        )

        sp_buf = generate_buffer(args, env, device)
        xp_buf0 = [generate_buffer(args, env, device) for _ in range(agent_num)]
        xp_buf1 = [generate_buffer(args, env, device) for _ in range(agent_num)]
        mp_buf = generate_buffer(args, env_mp, device, args.env_length - 1)

        if agent_num == N:
            run_dir = Path(base_dir + "/oracle_" + str(N) + ("_r" if true_recurrence else ""))
        else:
            run_dir = Path(base_dir + "/convention" + str(agent_num))

        config = {
            "all_args": args,
            "envs": env,
            "envs_mp": env_mp,
            "device": device,
            "num_agents": 2,
            "run_dir": run_dir,
        }
        print(run_dir)

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
        if agent_num < N:
            runner.model_dir = runner.save_dir
            policy_actor_state_dict = torch.load(str(runner.model_dir) + "/actor.pt")
            runner.policy.actor.load_state_dict(policy_actor_state_dict)
            print("restored", agent_num)
        else:
            runner.run()
        agent_set.append(next_agent.actor)
        

args = get_config().parse_args()
device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'

set_seed(args.seed, args.cuda_deterministic)
print(args)

pop_size = args.pop_size
env_generator = lambda x : generate_env(args.env_name, x, args.over_layout, use_env_cpu=(device=='cpu'))

args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name

if args.loss_type:
    run_dir = (
            os.path.dirname(os.path.abspath(__file__))
            + "/results/"
            + args.hanabi_name
            + "/baselines/"
            + args.loss_type
            + "/"
            + (args.run_dir)
            + "/"
            + str(args.seed)
        )
else:
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





# import os
# from config import get_config
# import torch
# import numpy as np
# import random

# from pathlib import Path

# from env_utils import generate_env, set_seed

# from BestResponse.oracle_player import OraclePlayer
# from XD.MCPolicy import MCPolicy
# from XD.xd_player import XDPlayer

# from MAPPO.utils.shared_buffer import SharedReplayBuffer
# from MAPPO.r_actor_critic import R_Actor


# def set_rands(seed):
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)

# def get_dir(args):
#     if args.loss_type:
#         return (
#             os.path.dirname(os.path.abspath(__file__))
#             + "/results/"
#             + args.hanabi_name
#             + "/baselines/"
#             + args.loss_type
#             + "/"
#             + (args.run_dir)
#             + "/"
#             + str(args.seed)
#         )
#     else:
#         return (
#             os.path.dirname(os.path.abspath(__file__))
#             + "/results/"
#             + args.hanabi_name
#             + "/"
#             + (args.run_dir)
#             + "/"
#             + str(args.seed)
#         )

# def main():
#     args = get_config().parse_args()
#     print(args)
#     pop_size = args.pop_size
#     device = "cuda" if torch.cuda.is_available() and args.cuda else 'cpu'
#     env = generate_env(args.env_name, args.n_rollout_threads * (pop_size * 2 + 1), args.over_layout, use_env_cpu=(device=='cpu'))

#     args.hanabi_name = args.over_layout if args.env_name == 'overcooked' else args.env_name
    
#     base_dir = get_dir(args)
#     os.makedirs(base_dir, exist_ok=True)

#     agent_set = []
#     set_rands(args.seed)
#     true_recurrence = args.use_naive_recurrent_policy
#     args.use_naive_recurrent_policy = False
#     for agent_num in range(pop_size):
#         actor = R_Actor(args, env.observation_space, env.action_space)
#         state_dict = torch.load(base_dir + "/convention" + str(agent_num) +"/models/actor.pt")
#         actor.load_state_dict(state_dict)
#         agent_set.append(actor)
#     args.use_naive_recurrent_policy = true_recurrence

#     run_dir = Path(base_dir + "/oracle_" + str(pop_size) + ("_r" if true_recurrence else ""))
#     config = {
#         "all_args": args,
#         "envs": env,
#         "device": device,
#         "num_agents": 2,
#         "run_dir": run_dir,
#     }
#     ego = OraclePlayer(config, agent_set)
#     ego.run()


# if __name__ == "__main__":
#     main()
