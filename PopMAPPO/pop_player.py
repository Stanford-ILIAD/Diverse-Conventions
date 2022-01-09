import time

from .pop_mappo import Pop_MAPPO
# from multiprocessing import Pool

# from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
# from MAPPO.main_player import MainPlayer
pla = []


def collect(player):
    print("DOING", player)
    pla[player].collect_episode()
    print("FINISH COLLECTING")
    return pla[player]


class PopPlayer:
    def __init__(self, args, players, pop_loss):
        self.players = players
        global pla
        pla = self.players

        policies = [player.policy for player in players]
        self.trainer = Pop_MAPPO(args, policies, pop_loss, popsize=len(players))
        self.pop_loss = pop_loss

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
            # with Pool() as p:
            #     global pla
            #     pla = p.map(collect, list(range(len(self.players))))
            #     self.players = pla
            for player in self.players:
                player.collect_episode()
            # for player in range(len(self.players)):
            #     collect(player)
            print("END POOL")

            for idx, player in enumerate(self.players):
                # player.collect_episode()
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
        train_infos = self.trainer.train(buffers)
        for player in self.players:
            player.buffer.chooseafter_update()
        # train_infos = [None] * len(self.players)
        # for idx, player in enumerate(self.players):
        #     train_infos[idx] = player.train()
        return train_infos
