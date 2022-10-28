import time

from .pop_mappo import Pop_MAPPO


class PopPlayer:
    def __init__(self, args, players, pop_loss):
        self.players = players

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
            for player in self.players:
                player.collect_episode()
            print("END POOL")

            for idx, player in enumerate(self.players):
                total_num_steps[idx] += player.episode_length

                print(f"Player {idx}")
                # post process
                player.log(train_infos[idx], episode, episodes, total_num_steps[idx], start)

                # compute return and update network
                player.compute()
                if player.use_linear_lr_decay:
                    print(idx)
                    self.trainer.policies[idx].lr_decay(episode, episodes)

            train_infos = self.train()
            print("DONE TRAINING:", episode)

    def train(self):
        self.trainer.prep_training()
        buffers = [player.buffer for player in self.players]
        train_infos = self.trainer.train(buffers)
        for player in self.players:
            player.buffer.chooseafter_update()
        return train_infos
