import sys
import torch

from tree_env.tree_env import DecentralizedTree
from tree_env.tree_agent import (TreeUser, RandomTreeAgent, SafeAgent,
                                 RiskyAgent)
from partner_agents import DecentralizedAgent
from MAPPO.r_actor_critic import R_Actor
from config import get_config

EGO_LIST = ['RAND', 'SAFE', 'RISKY', 'LOAD']

def gen_agent(value, env, args=None):
    if value == 'RAND':
        return RandomTreeAgent()
    if value == 'SAFE':
        return SafeAgent()
    if value == 'RISKY':
        return RiskyAgent()
    if value == 'LOAD':
        actor = R_Actor(args, env.observation_space, env.action_space)
        print(args.partner_load)
        if args.partner_load is None:
            print("NEED TO INPUT FILE")
            sys.exit()
        state_dict = torch.load(args.partner_load)
        actor.load_state_dict(state_dict)
        return DecentralizedAgent(actor)

def run_sim(env, ego, args):
    env.add_partner_agent(TreeUser(env))
    env.ego_ind = args.agent_ind
    game = 0
    while True:
        print(f'Game #{game}')
        obs = env.reset()
        done = False
        reward = 0
        while not done:
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward
        print('Reward is', reward)
        print()
        should_continue = -1
        while should_continue == -1:
            i = input('Continue [(y)es/(n)o]? ').lower()
            if i in ('y', 'yes'):
                should_continue = 1
            elif i in ('n', 'no'):
                should_continue = 0
            else:
                print('INVALID INPUT')
        if not should_continue:
            print('Bye!')
            return
        game += 1

def main(parser):
    env = DecentralizedTree()
    args = parser.parse_args()
    ego = gen_agent(args.partner, env, args)
    run_sim(env, ego, args)

if __name__ == '__main__':
    parser = get_config()
    parser.add_argument("--agent_ind", type=int, default=1,
                        help="Index of partner agent")
    parser.add_argument('partner',
                        choices=EGO_LIST,
                        help='Algorithm for the partner agent')
    parser.add_argument('--partner-load',
                        help='File to load the partner agent from')
    main(parser)
