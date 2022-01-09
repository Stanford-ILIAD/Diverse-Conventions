from hanabi_env.hanabi_env import MaskedHanabi
from MAPPO.main_player import MainPlayer
from PopMAPPO.pop_player import PopPlayer
from DMAPPO.poploss import PopulationLoss, ADAPLoss
from hanabi_agent import CentralizedAgent

from config import get_config
import os
from pathlib import Path

POPULATION_SIZE = 3


def generate_player(args):
    args.hanabi_name = 'MaskedHanabi'
    args.n_rollout_threads = 1
    args.episode_length = 200
    args.ppo_epoch = 15
    args.gain = 0.01
    args.lr = 7e-4
    args.critic_lr = 1e-3
    args.hidden_size = 512
    args.layer_N = 2
    args.entropy_coef = 0.015
    args.use_recurrent_policy = False
    args.use_value_active_masks = False
    args.use_policy_active_masks = False

    han_config={
                "colors":
                    2,
                "ranks":
                    5,
                "players":
                    2,
                "hand_size":
                    2,
                "max_information_tokens":
                    3,
                "max_life_tokens":
                    1,
                "observation_type":1
            }
    env = MaskedHanabi(han_config)
    print(env.observation_space)
    print(env.share_observation_space)
    print(env.action_space)
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
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


def main():
    args = get_config().parse_args()
    players = [generate_player(args) for _ in range(POPULATION_SIZE)]
    pop_runner = PopPlayer(args, players, ADAPLoss(0.2))
    pop_runner.run()


if __name__ == '__main__':
    main()
