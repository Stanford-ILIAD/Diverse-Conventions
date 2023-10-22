import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np

from config import get_config

"""
{
  convention0: {
    1: {0.0: [all scores], 1.0: ..., 3.0: ...}
    2: ...
  }
}
"""

# Colors: Orange: ff9b00
# Blue: 2a6494
COLORS = [
    '#ff9b00',  # orange
    '#2a6494',  # blue
    '#595958'  # black
]

LABELS = ['G', 'S', '0']

matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['text.color'] = 'black'


def read_data(txt_file):
    output = {}
    with open(txt_file) as f:
        for line in f:
            split_line = line.strip().split(",")
            # first section is always episode
            epnum = int(split_line[0].split(":")[1])
            split_line = split_line[1:]
            temp_dict = {}
            for x in split_line:
                k, v = tuple(x.split(":"))
                temp_dict[float(k)] = float(v)

            total_episodes = sum(temp_dict.values())
            line_data = {k: v/total_episodes for k, v in temp_dict.items()}
            output[epnum] = line_data
    return output


def make_uniform(seed_dict):
    score_set = set()
    for seed in seed_dict:
        for conv in seed_dict[seed]:
            for epnum in seed_dict[seed][conv]:
                score_set.update(set(seed_dict[seed][conv][epnum].keys()))
    print(score_set)
    for seed in seed_dict:
        for conv in seed_dict[seed]:
            for epnum in seed_dict[seed][conv]:
                for score in score_set:
                    if score not in seed_dict[seed][conv][epnum].keys():
                        seed_dict[seed][conv][epnum][score] = 0.0
    return score_set


def restructure_data(seed_dict):
    output = {}
    for seed in seed_dict:
        for conv in seed_dict[seed]:
            if conv not in output:
                output[conv] = {}
            for epnum in seed_dict[seed][conv]:
                if epnum not in output[conv]:
                    output[conv][epnum] = {}
                for score, rate in seed_dict[seed][conv][epnum].items():
                    if score not in output[conv][epnum]:
                        output[conv][epnum][score] = []
                    output[conv][epnum][score].append(rate)
    return output


def make_plot(name, convention, data, score_set):
    xdata = list(data.keys())
    # print(convention)
    # print(xdata)
    ydata_mean = {v: [] for v in score_set}
    ydata_min = {v: [] for v in score_set}
    ydata_max = {v: [] for v in score_set}

    for episode in sorted(data):
        # print(episode)
        for score in data[episode]:
            nparr = np.array(data[episode][score])
            ydata_min[score].append(nparr.min())
            ydata_max[score].append(nparr.max())
            ydata_mean[score].append(nparr.mean())

    # print(ydata_min)

    # print(ydata_mean)

    # print(ydata_max)
    # print("-" * 50)
    plt.clf()
    # plt.title(f"{name} Convention {convention}", fontsize=20)
    plt.xlabel("Training Epoch", color="black", fontsize=30)
    plt.ylabel("Frequency", color="black", fontsize=30)
    for i, score in enumerate([3.0, 1.0, 0.0]):
        plt.plot(xdata,
                 ydata_mean[score],
                 label=LABELS[i],
                 linewidth=5.0,
                 color=COLORS[i])
        plt.fill_between(xdata,
                         ydata_min[score],
                         ydata_max[score],
                         alpha=0.2,
                         color=COLORS[i])
    plt.legend(frameon=False, loc=7, fontsize=30)
    for pos in ['right', 'top']:
        plt.gca().spines[pos].set_visible(False)
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')

    plt.gca().tick_params(axis='x', colors='black')
    plt.gca().tick_params(axis='y', colors='black')

    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    plt.tight_layout()

    plt.savefig(f"{name}_convention{convention}.pdf")
    plt.show()


def main(parser):
    args = parser.parse_args()
    args.hanabi_name = args.env_name
    middle_dir = "/results/"
    if args.loss_type is not None:
        middle_dir = f"/baselines/{args.loss_type}/"
    base_dir = (
        os.path.dirname(os.path.abspath(__file__))
        + "/../" + ("Validation/" if args.do_validation else "")
        + args.hanabi_name
        + middle_dir
        + (args.run_dir)
    )

    results_seed_convention = {}
    for seed in os.listdir(base_dir):
        run_dir = base_dir + "/" + seed
        if not os.path.isdir(run_dir):
            continue
        # print(seed)
        seed_ans = {}
        for convention in os.listdir(run_dir):
            if not os.path.isdir(run_dir + "/" + convention):
                continue
            convention_log = run_dir + "/" + convention + "/logs/sp.txt"
            seed_ans[int(convention[10:])] = read_data(convention_log)
        results_seed_convention[seed] = seed_ans

    score_set = make_uniform(results_seed_convention)

    full_data = restructure_data(results_seed_convention)

    for convention in full_data:
        make_plot(args.name, str(convention) + ("_VALIDATION" if args.do_validation else ""), full_data[convention], score_set)


if __name__ == "__main__":
    parser = get_config()
    parser.add_argument('--name', type=str, help="Name of algorithm")
    main(parser)
