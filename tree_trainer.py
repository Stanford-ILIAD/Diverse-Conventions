from tree_env.tree_env import PantheonTree
from MAPPO.main_player import MainPlayer
from hanabi_agent import CentralizedAgent
from PopMAPPO.pop_player import PopPlayer
from DMAPPO.poploss import PopulationLoss, ADAPLoss

from config import get_config
import os
from pathlib import Path

args = get_config().parse_args()
args.hanabi_name = 'Tree'
print(args)

env = PantheonTree()
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
ego.run()

# def get_loss(args):
#     if args.loss_type is None:
#         return PopulationLoss()
#     elif args.loss_type == 'ADAP':
#         return ADAPLoss(args.loss_param)
#     else:
#         print("Invalid Loss Type; Assuming no loss")
#         return PopulationLoss()


# def generate_player(args):
#     args.hanabi_name = 'Tree'
#     env = PantheonTree()
#     run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/tree/results/" + (args.loss_type or "standard") + "/" + str(args.seed))

#     print(run_dir)
#     config = {
#         'all_args': args,
#         'env': env,
#         'device': 'cpu',
#         'num_agents': 2,
#         'run_dir': run_dir
#     }
#     ego = MainPlayer(config)
#     partner = CentralizedAgent(ego, 1)
#     env.add_partner_agent(partner)
#     args.seed += 100
#     return ego


# def main():
#     args = get_config().parse_args()
#     print(args)
#     population_size = args.pop_size
#     players = [generate_player(args) for _ in range(population_size)]
#     pop_runner = PopPlayer(args, players, get_loss(args))
#     pop_runner.run()


# if __name__ == '__main__':
#     main()
