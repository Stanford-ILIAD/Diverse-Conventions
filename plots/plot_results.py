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


def make_plot(convention, data, score_set):
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
    plt.title(f"Frequency of Scores for XD Convention {convention}")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Frequency of Score")
    for score in score_set:
        plt.plot(xdata, ydata_mean[score], label=f"Score={score}")
        plt.fill_between(xdata, ydata_min[score], ydata_max[score], alpha=0.2)
    plt.legend()
    plt.savefig(f"xd_convention{convention}.png")
    plt.show()
    
def main(parser):
    args = parser.parse_args()
    args.hanabi_name = args.env_name
    base_dir = (
        os.path.dirname(os.path.abspath(__file__))
        + "/"
        + args.hanabi_name
        # + "/baselines/ADAP/"
        + "/results/"
        + (args.run_dir)
    )

    results_seed_convention = {}
    for seed in os.listdir(base_dir):
        # print(seed)
        run_dir = base_dir + "/" + seed
        seed_ans = {}
        for convention in os.listdir(run_dir):
            if convention == "args.txt":
                continue
            convention_log = run_dir + "/" + convention + "/logs/sp.txt"
            seed_ans[int(convention[10:])] = read_data(convention_log)
        results_seed_convention[seed] = seed_ans
        
    score_set = make_uniform(results_seed_convention)

    full_data = restructure_data(results_seed_convention)
    # print(full_data)

    for convention in full_data:
        make_plot(convention, full_data[convention], score_set)
    # plt.show()

if __name__ == "__main__":
    parser = get_config()

    main(parser)
