import time
from itertools import combinations

from .xmappo import XMAPPO
from hanabi_agent import CentralizedAgent


class XPlayer:
    def __init__(self, args, players, xgym, xp_buffers, xlength, alpha):
        self.players = players
        self.alpha = alpha

        policies = [player.policy for player in players]
        self.trainer = XMAPPO(args, policies, popsize=len(players))

        self.xgym = xgym

        self.xlength = xlength

        N = len(players)
        self.xp_buffers = []
        for i in range(N):
            self.xp_buffers.append([])
            for j in range(i):
                self.xp_buffers[i].append(xp_buffers[j][i-j-1])
            for j in range(N-i-1):
                self.xp_buffers[i].append(xp_buffers[i][j])

    def collect_cross(self, p1, p2):
        player1 = self.players[p1]
        player2 = self.players[p2]
        partner = CentralizedAgent(player1, 1, player2.policy)
        self.xgym.add_partner_agent(partner)

        buffer = self.xp_buffers[p1][p2-1]

        origenv = player1.env
        player1.env = self.xgym
        player1.collect_episode(buffer, self.xlength, False)

        self.xgym.partners[0].clear()
        player1.env = origenv

    def run(self):
        for player in self.players:
            player.setup_data()
            player.warmup()

        start = time.time()
        episodes = int(self.players[0].num_env_steps) // self.players[0].episode_length
        train_infos = [None] * len(self.players)
        total_num_steps = [0] * len(self.players)

        for episode in range(episodes):
            print("START POOL")
            for player in self.players:
                player.collect_episode()

            for i, j in combinations(range(len(self.players)), 2):
                self.collect_cross(i, j)
            print("END POOL")

            for idx, player in enumerate(self.players):
                total_num_steps[idx] += player.episode_length

                print(f"Player {idx}")
                # post process
                player.log(train_infos[idx], episode, episodes, total_num_steps[idx], start)

                # compute return and update network
                player.compute()

            train_infos = self.train()
            print("DONE TRAINING:", episode)

    def train(self):
        self.trainer.prep_training()
        buffers = [player.buffer for player in self.players]
        train_infos = self.trainer.train(buffers, self.xp_buffers, self.alpha)
        for player in self.players:
            player.buffer.chooseafter_update()
        return train_infos
