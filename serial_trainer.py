import os
from config import get_config

from tree_env.tree_env import PantheonTree
from numline_env.numline_env import PantheonLine
from hanabi_env.hanabi_env import MaskedHanabi
from overcooked_env.overcooked_env import PantheonOvercooked
from XD.serial import run_serial


def generate_gym(args):
    """Generate the gym given the command-line arguments."""
    if args.env_name == "Tree":
        args.hanabi_name = "Tree"
        return PantheonTree()
    if args.env_name == "Line":
        args.hanabi_name = "Line"
        return PantheonLine()
    if args.env_name == "Overcooked":
        args.hanabi_name = "Overcooked"
        return PantheonOvercooked(args.over_layout)
    if args.env_name == "Hanabi":
        han_config = {
            "colors": args.han_colors,
            "ranks": args.han_ranks,
            "players": 2,
            "hand_size": args.han_hand,
            "max_information_tokens": args.han_info,
            "max_life_tokens": args.han_life,
            "observation_type": 1,
        }
        return MaskedHanabi(han_config)
    return None


def main():
    args = get_config().parse_args()
    print(args)
    pop_size = args.pop_size
    env = generate_gym(args)
    run_dir = (
        os.path.dirname(os.path.abspath(__file__))
        + "/" + ("Validation/" if args.do_validation else "")
        + args.hanabi_name
        + "/results/"
        + (args.run_dir)
        + "/"
        + str(args.seed)
    )
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir + "/" + "args.txt", "w", encoding="UTF-8") as file:
        file.write(str(args))

    run_serial(pop_size, args, env, run_dir, "cpu", restored=args.restored)


if __name__ == "__main__":
    main()
