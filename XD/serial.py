from .MCPolicy import MCPolicy
from .xd_player import XDPlayer
from MAPPO.utils.shared_buffer import SharedReplayBuffer

import torch
import numpy as np
import random
from pathlib import Path


def set_rands(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_buffer(args, env):
    return SharedReplayBuffer(
        args, 2, env.observation_space, env.share_observation_space, env.action_space
    )


def run_serial(N, args, env, base_dir, device, restored=0):
    print(N, "agents total")
    agent_set = []
    for agent_num in range(N):
        set_rands(args.seed + args.seed_skip * agent_num)
        print("Training agent", agent_num)
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
        if agent_num < restored:
            runner.model_dir = runner.save_dir
            runner.restore()
            print("restored", agent_num)
        else:
            runner.run()

        agent_set.append(next_agent.actor)
