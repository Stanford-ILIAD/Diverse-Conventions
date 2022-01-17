from hanabi_env.hanabi_env import MaskedHanabi
from MAPPO.main_player import MainPlayer
from PopMAPPO.pop_player import PopPlayer
from DMAPPO.poploss import PopulationLoss, ADAPLoss
from hanabi_agent import CentralizedAgent

from config import get_config
import os
from pathlib import Path

def get_loss(args):
    if args.loss_type is None:
        return PopulationLoss()
    elif args.loss_type == 'ADAP':
        return ADAPLoss(args.loss_param)
    else:
        print("Invalid Loss Type; Assuming no loss")
        return PopulationLoss()


def generate_player(args):
    args.hanabi_name = 'MaskedHanabi'

    han_config={
                "colors":
                    args.han_colors,
                "ranks":
                    args.han_ranks,
                "players":
                    2,
                "hand_size":
                    args.han_hand,
                "max_information_tokens":
                    args.han_info,
                "max_life_tokens":
                    args.han_life,
                "observation_type":1
            }
    env = MaskedHanabi(han_config)
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
    args.seed += 100
    return ego


def main():
    args = get_config().parse_args()
    print(args)
    population_size = args.pop_size
    players = [generate_player(args) for _ in range(population_size)]
    pop_runner = PopPlayer(args, players, get_loss(args))
    pop_runner.run()


if __name__ == '__main__':
    main()
