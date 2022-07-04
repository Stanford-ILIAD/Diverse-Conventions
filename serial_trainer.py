from tree_env.tree_env import PantheonTree
from numline_env.numline_env import PantheonLine
from XD.serial import run_serial

from config import get_config
import os

def generate_gym(args):
    if args.env_name == 'Tree':
        args.hanabi_name = "Tree"
        return PantheonTree()
    elif args.env_name == 'Line':
        args.hanabi_name = 'Line'
        return PantheonLine()
    elif args.env_name == 'Hanabi':
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


def main():
    args = get_config().parse_args()
    print(args)
    N = args.pop_size
    env = generate_gym(args)
    run_dir = (
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        + "/"
        + args.hanabi_name
        + "/results/"
        + (args.loss_type or "standard")
        + "/"
        + str(args.seed)
    )
    run_serial(N, args, env, run_dir, "cpu")


if __name__ == "__main__":
    main()
