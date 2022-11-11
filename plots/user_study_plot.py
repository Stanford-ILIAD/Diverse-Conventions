from config import get_config

from collections import defaultdict
import statistics
from math import sqrt

import os
import json

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCORES = defaultdict(lambda: defaultdict(lambda: []))
FIRST_DELIVERY = defaultdict(lambda: defaultdict(lambda: []))
WEIGHTED_SCORES = defaultdict(lambda: defaultdict(lambda: []))

ALGO_LIST = ['SP', 'ADAP', 'XP', 'MP']
ALGO_COLORS = [
    mcolors.CSS4_COLORS['lightgray'],
    mcolors.CSS4_COLORS['gray'],
    mcolors.CSS4_COLORS['khaki'],
    mcolors.CSS4_COLORS['darkorange']
]

LAYOUT_DICT = {
    "cramped_room": "Cramped Room",
    "coordination_ring": "Coordination Ring"
}

USE_BEST = True

MAX_TIMESTEPS = 203


def get_stdev(values):
    return statistics.pstdev(values) / sqrt(len(values))


def insert_traj(layout, algo, timespaces):
    if len(timespaces) == 0:
        return
    SCORES[layout][algo] += [len(timespaces)]
    if len(timespaces) > 1:
        FIRST_DELIVERY[layout][algo] += [timespaces[0] + timespaces[1]]

    weighted_score = 0
    for i in range(len(timespaces)):
        weighted_score += MAX_TIMESTEPS - timespaces[i]
    weighted_score = timespaces[-1] / len(timespaces)
    WEIGHTED_SCORES[layout][algo].append(weighted_score)


def parse_traj(path, entrynum, layout, algo):
    if entrynum >= 2:
        return []
    trajectory = None
    with open(path) as f:
        trajectory = json.load(f)

    if trajectory is None:
        return []

    rewards = trajectory['ep_rewards'][0]
    if len(rewards) == 0:
        return []
    print(len(rewards))
    serveindex = [i for i, score in enumerate(rewards) if score == 20]
    print(f"{serveindex} from {path}")
    serveindex = [0] + serveindex

    timespaces = [
        serveindex[i+1] - serveindex[i] for i in range(len(serveindex)-1)
    ]
    # print(timespaces)

    # SCORES[layout][algo] += [len(timespaces)]

    # if len(timespaces) > 1:
    #     FIRST_DELIVERY[layout][algo] += [timespaces[0] + timespaces[1]]
    if not USE_BEST:
        insert_traj(layout, algo, timespaces)
    return timespaces


def parse_config(path, layout, algo):
    for id in os.listdir(path):
        iddir = path + "/" + id

        if id == 'dorsa' or id == 'test0' or id == 'andyshih':
            continue
        # if id != 'brianxu':
        #     continue

        if not os.path.isdir(iddir):
            continue

        for ip in os.listdir(iddir):
            ipdir = iddir + "/" + ip
            if not os.path.isdir(ipdir):
                continue

            besttimespaces = []
            # print(f"User {id} with ip {ip}")
            for entry in os.listdir(ipdir):
                splitentry = entry.split('.')
                if splitentry[-1] == 'json':
                    # print(f"Parsing {entry}")
                    timespaces = parse_traj(
                        ipdir + "/" + entry,
                        int(splitentry[0]),
                        layout,
                        algo
                    )
                    if len(timespaces) > len(besttimespaces):
                        besttimespaces = timespaces
            if USE_BEST:
                insert_traj(layout, algo, besttimespaces)


def parse_files():
    if ARGS.trajs_savepath[-1] == "/":
        ARGS.trajs_savepath = ARGS.trajs_savepath[:-1]
    path = ARGS.trajs_savepath
    for layout in os.listdir(path):
        layoutdir = path + "/" + layout
        if not os.path.isdir(layoutdir):
            continue

        for algo in os.listdir(layoutdir):
            algodir = layoutdir + "/" + algo
            if not os.path.isdir(algodir):
                continue

            # print(f"PARSING {layout} with {algo}")
            parse_config(algodir, layout, algo)


def plotScores():
    for key in SCORES:
        algo_scores = []
        algo_std = []
        for algo in ALGO_LIST:
            algo_scores.append(sum(SCORES[key][algo])/len(SCORES[key][algo]))
            algo_std.append(get_stdev(SCORES[key][algo]))
        plt.clf()
        plt.title(f"Scores in Layout {LAYOUT_DICT[key]}")
        plt.ylabel("Number of Dishes Served")
        plt.bar(ALGO_LIST, algo_scores, yerr=algo_std, color=ALGO_COLORS)
        plt.show()


def plotFirstTimes():
    for key in FIRST_DELIVERY:
        algo_scores = []
        algo_std = []
        for algo in ALGO_LIST:
            m = sum(FIRST_DELIVERY[key][algo])/len(FIRST_DELIVERY[key][algo])
            algo_std.append(get_stdev(FIRST_DELIVERY[key][algo]))
            algo_scores.append(m)
        plt.clf()
        plt.title(f"Time for Two Deliveries in Layout {LAYOUT_DICT[key]}")
        plt.ylabel("Number of Timesteps")
        plt.bar(ALGO_LIST, algo_scores, yerr=algo_std, color=ALGO_COLORS)
        plt.show()


def plotWeightedScores():
    for key in SCORES:
        algo_scores = []
        algo_std = []
        for algo in ALGO_LIST:
            m = sum(WEIGHTED_SCORES[key][algo])/len(WEIGHTED_SCORES[key][algo])
            algo_scores.append(m)
            algo_std.append(get_stdev(WEIGHTED_SCORES[key][algo]))
        plt.clf()
        plt.title(f"Average Time per Delivery in Layout {LAYOUT_DICT[key]}")
        plt.ylabel("Number of Timesteps")
        plt.bar(ALGO_LIST, algo_scores, yerr=algo_std, color=ALGO_COLORS)
        plt.show()


def main():
    parse_files()
    print(SCORES)
    print(FIRST_DELIVERY)

    plotScores()
    # plotFirstTimes()
    plotWeightedScores()


if __name__ == '__main__':
    parser = get_config()
    parser.add_argument('--trajs_savepath', type=str,
                        help="folder to save trajectories")
    ARGS = parser.parse_args()
    main()
