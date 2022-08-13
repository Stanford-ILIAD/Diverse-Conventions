import time
from collections import Counter
import os
import numpy as np

from MAPPO.main_player import MainPlayer
from MAPPO.rMAPPOPolicy import R_MAPPOPolicy
from MAPPO.utils.shared_buffer import SharedReplayBuffer

from partner_agents import CentralizedAgent

from .mappo import MAPPO


def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in sorted(Counter(x).items())])


class OraclePlayer(MainPlayer):
    def __init__(self, config, agent_set):
        self._init_vars(config)

        share_observation_space = self.env.share_observation_space

        # policy network
        self.policy = R_MAPPOPolicy(
            self.all_args,
            self.env.observation_space,
            share_observation_space,
            self.env.action_space,
            device=self.device,
        )

        if self.model_dir is not None:
            self.restore()

        # algorithm TODO
        self.trainer = MAPPO(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffers = [SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.env.observation_space,
            share_observation_space,
            self.env.action_space,
        ) for _ in range(2)]

        self.buffer = self.buffers[0]

        self.agent_set = agent_set
        
        self.env.partners[0].clear()
        for partner in agent_set:
            pagent = CentralizedAgent(self, 1, partner)
            self.env.add_partner_agent(pagent, 1)
        pagent = CentralizedAgent(self, 1)
        self.env.add_partner_agent(pagent, 1)

    def set_xp(self, ego_id):
        self.ego_id = ego_id
        self.env.ego_ind = ego_id

        self.buffer = self.buffers[ego_id]

        for partner in self.env.partners[0]:
            partner.player_id = 1 - self.ego_id

    def log(self, train_infos, episode, episodes, total_num_steps, start):
        # save model
        if episode % self.save_interval == 0 or episode == episodes - 1:
            self.save()

        if episode == 0:
            # Setup files
            files = []
            # log.txt
            # Env algo exp updates ... avg score, avg xp score
            files.append(self.log_dir + "/log.txt")

            # xp0.txt, xp1.txt
            # t: episode, Counter
            files.append(self.log_dir + "/xp0.txt")
            
            files.append(self.log_dir + "/xp1.txt")

            os.makedirs(self.log_dir, exist_ok=True)
            for file in files:
                with open(file, "w", encoding="UTF-8"):
                    pass

        # log information
        if train_infos is not None or (
            episode % self.log_interval == 0 and episode > 0
        ):
            end = time.time()
            files = {}

            avg0 = np.mean(self.scores0)
            avg1 = np.mean(self.scores1)
            average_score = (avg0 + avg1) / 2

            general_log = (
                f"Updates:{episode}/{episodes},"
                + f"Timesteps:{total_num_steps}/{self.num_env_steps},"
                + f"FPS:{total_num_steps//(end-start)},"
                + f"avg_score:{average_score}"
            )

            print(
                "\n Env {} Algo {} Exp {} updates {}/{} episodes, \
                total num timesteps {}/{}, FPS {}.\n".format(
                    self.all_args.hanabi_name,
                    self.algorithm_name,
                    self.experiment_name,
                    episode,
                    episodes,
                    total_num_steps,
                    self.num_env_steps,
                    int(total_num_steps / (end - start)),
                )
            )

            print("average score is {}.".format(average_score))

            general_log += f",avg_xp0:{avg0},avg_xp1:{avg1}"
            print(f"average xp score as 0 is {avg0}.")
            print(f"average xp score as 1 is {avg1}.")

            # self.log_train(train_infos, self.true_total_num_steps)
            print(train_infos)
            general_log += "," + ",".join(
                [f"{key}:{val}" for key, val in train_infos.items()]
            )

            files["log.txt"] = general_log

            files["xp0.txt"] = get_histogram(self.scores0)
            files["xp1.txt"] = get_histogram(self.scores1)
            print("Cross-play Scores counts (ego id 0): ", files["xp0.txt"])
            print("Cross-play Scores counts (ego id 1): ", files["xp1.txt"])
            
            for key, val in files.items():
                with open(f"{self.log_dir}/{key}", "a", encoding="UTF-8") as file:
                    file.write(f"episode:{episode},{val}\n")

    def run(self):
        self.setup_data()
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )
        train_infos = None
        total_num_steps = 0

        self.best_i = None

        for episode in range(episodes):
            self.set_xp(0)
            self.collect_episode(turn_based=not self.all_args.simul_env)
            self.scores0 = self.scores

            self.set_xp(1)
            self.collect_episode(turn_based=not self.all_args.simul_env)
            self.scores1 = self.scores

            total_num_steps += self.episode_length
            # post process
            self.log(train_infos, episode, episodes, total_num_steps, start)

            # compute return and update network
            self.compute_all()
            train_infos = self.train()
            print("DONE TRAINING:", episode)

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffers[0], self.buffers[1])
        for buf in self.buffers:
            buf.reset_after_update()
        return train_infos

    def compute_all(self):
        self.set_xp(0)
        self.compute()

        self.set_xp(1)
        self.compute()
