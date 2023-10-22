import time
from collections import Counter
import os
import torch
import numpy as np

from MAPPO.main_player import MainPlayer
from MAPPO.rMAPPOPolicy import R_MAPPOPolicy, rMAPPOWrapper
from MAPPO.utils.shared_buffer import SharedReplayBuffer

from partner_agents import CentralizedMultiAgent

from .mappo import MAPPO


def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in sorted(Counter(x).items())])


class OraclePlayer(MainPlayer):
    def __init__(self, config, agent_set):
        self._init_vars(config)

        share_observation_space = self.envs.share_observation_space

        # policy network
        self.policy = R_MAPPOPolicy(
            self.all_args,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            device=self.device,
        )

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = MAPPO(self.all_args, self.policy, device=self.device)

        # buffer
        # self.buffers = [ for _ in range(3)]

        self.sp_buf = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            self.device
        )
        self.xp_buf0 = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            self.device,
            self.all_args.n_rollout_threads * len(agent_set)
        )
        self.xp_buf1 = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            self.device,
            self.all_args.n_rollout_threads * len(agent_set)
        )

        self.buffers = [self.sp_buf, self.xp_buf0, self.xp_buf1]
        
        self.buffer = self.buffers[0]

        self.agent_set = agent_set

        self.envs.partners[0].clear()
        wrapped_agent_set = [rMAPPOWrapper(x, self.policy.critic) for i, x in enumerate(self.agent_set)]
        
        policies = [self.policy] + wrapped_agent_set + [self.policy] * (len(agent_set))
        self.envs.add_partner_agent(CentralizedMultiAgent(self, 1-self.ego_id, policies, self.all_args.n_rollout_threads))

    def collect_episode(self, buffer=None, length=None, save_scores=True): # REDO
        self.running_score = torch.zeros((self.envs.num_envs), device=self.device)

        if length is None:
            length = self.episode_length

        if save_scores:
            self.scores = [[] for _ in range(len(self.agent_set) * 2 + 1)]

        wrapped_agent_set = [rMAPPOWrapper(x, self.policy.critic) for i, x in enumerate(self.agent_set)]

        use_policies = [self.policy] * (len(self.agent_set) + 1) + wrapped_agent_set

        threads = self.all_args.n_rollout_threads

        p = len(self.agent_set)
            
        for _ in range(length):
            self.next_step(use_policies, threads, self.scores if save_scores else None)
            self.sp_buf.chooseinsert(self.turn_share_obs[:threads],
                                     self.turn_obs[:threads],
                                     self.turn_rnn_states[:threads],
                                     self.turn_rnn_states_critic[:threads],
                                     self.turn_actions[:threads],
                                     self.turn_action_log_probs[:threads],
                                     self.turn_values[:threads],
                                     self.turn_rewards[:threads],
                                     self.turn_masks[:threads],
                                     self.turn_bad_masks[:threads],
                                     self.turn_active_masks[:threads],
                                     self.turn_available_actions[:threads])

            # for i in range(len(self.agent_set)):
            self.xp_buf0.chooseinsert(self.turn_share_obs[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_obs[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_rnn_states[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_rnn_states_critic[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_actions[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_action_log_probs[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_values[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_rewards[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_masks[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_bad_masks[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_active_masks[threads : threads * (len(self.agent_set) + 1)],
                                     self.turn_available_actions[threads : threads * (len(self.agent_set) + 1)])

            self.xp_buf1.chooseinsert(self.turn_share_obs[threads * (len(self.agent_set) + 1) :],
                                     self.turn_obs[threads * (len(self.agent_set) + 1) :],
                                     self.turn_rnn_states[threads * (len(self.agent_set) + 1) :],
                                     self.turn_rnn_states_critic[threads * (len(self.agent_set) + 1) :],
                                     self.turn_actions[threads * (len(self.agent_set) + 1) :],
                                     self.turn_action_log_probs[threads * (len(self.agent_set) + 1) :],
                                     self.turn_values[threads * (len(self.agent_set) + 1) :],
                                     self.turn_rewards[threads * (len(self.agent_set) + 1) :],
                                     self.turn_masks[threads * (len(self.agent_set) + 1) :],
                                     self.turn_bad_masks[threads * (len(self.agent_set) + 1) :],
                                     self.turn_active_masks[threads * (len(self.agent_set) + 1) :],
                                     self.turn_available_actions[threads * (len(self.agent_set) + 1) :])
            

        self.sp_scores = self.scores[0]
        self.xp_scores = [[self.scores[1 + i + p * j] for i in range(len(self.agent_set))] for j in range(2)]
        # accumulated_scores = [[sum(x) / len(x) for x in s] for s in self.xp_scores]
        # full_accumulated_scores = [accumulated_scores[0][i] + accumulated_scores[1][i] for i in range(p)]
        # if len(full_accumulated_scores) > 0:
        #     self.best_i = full_accumulated_scores.index(max(full_accumulated_scores))
        #     print("best i is", self.best_i, "because", full_accumulated_scores)
        # # mp handled separately

        # if self.mp_weight > 0 and len(self.agent_set) > 0:
        #     self.collect_mp_episode(length)
        
        # buffer = buffer or self.buffer
        # self.running_score = torch.zeros((self.n_rollout_threads), device=self.device)
        # if length is None:
        #     length = self.episode_length
        # if save_scores:
        #     self.scores = []

        # for _ in range(length):
        #     self.next_step(self.scores if save_scores else None)
        #     self.buffer.chooseinsert(self.turn_share_obs,
        #                              self.turn_obs,
        #                              self.turn_rnn_states,
        #                              self.turn_rnn_states_critic,
        #                              self.turn_actions,
        #                              self.turn_action_log_probs,
        #                              self.turn_values,
        #                              self.turn_rewards,
        #                              self.turn_masks,
        #                              self.turn_bad_masks,
        #                              self.turn_active_masks,
        #                              self.turn_available_actions)

    @torch.no_grad()
    def next_step(self, use_policies, threads, scores=None): # REDO
        self.trainer.prep_rollout()

        # print(use_policies)
        # print(self.use_obs.shape)
        # print(self.turn_obs.shape)
        self.turn_obs[:, self.ego_id] = self.use_obs
        self.turn_share_obs[:, self.ego_id] = self.use_share_obs
        self.turn_available_actions[:, self.ego_id] = self.use_available_actions
        self.turn_active_masks[:, 1 - self.ego_id] = 0
        self.turn_active_masks[:, self.ego_id] = 1

        for i, p in enumerate(use_policies):
            # print("TURN ACTIONS", self.turn_actions.flatten())
            x = p.get_actions(
                self.use_share_obs[i * threads : (i + 1) * threads],
                self.use_obs[i * threads : (i + 1) * threads],
                self.turn_rnn_states[i * threads : (i + 1) * threads, self.ego_id],
                self.turn_rnn_states_critic[i * threads : (i + 1) * threads, self.ego_id],
                self.turn_masks[i * threads : (i + 1) * threads, self.ego_id],
                self.use_available_actions[i * threads : (i + 1) * threads]
            )
            # print(x[1].flatten())
            (
                self.turn_values[i * threads : (i + 1) * threads, self.ego_id],
                self.turn_actions[i * threads : (i + 1) * threads, self.ego_id],
                self.turn_action_log_probs[i * threads : (i + 1) * threads, self.ego_id],
                self.turn_rnn_states[i * threads : (i + 1) * threads, self.ego_id],
                self.turn_rnn_states_critic[i * threads : (i + 1) * threads, self.ego_id]
            ) = x

        vobs, rewards, done, info = self.envs.step(
            self.turn_actions[:, self.ego_id]
        )
        dones = done.to(torch.bool)
        self.use_obs = vobs.obs.clone()
        self.use_share_obs = vobs.state.clone()
        self.use_available_actions = vobs.action_mask.clone()
        self.turn_rewards[:, self.ego_id] = rewards[:, None]

        # print(self.running_score, rewards)
        self.running_score += rewards

        self.turn_masks[dones, self.ego_id] = 0
        self.turn_rnn_states[dones, self.ego_id] = 0
        self.turn_rnn_states_critic[dones, self.ego_id] = 0

        self.turn_masks[~dones, self.ego_id] = 1

        if scores is not None and torch.any(dones):
            for i in range(len(use_policies)):
                scores[i].extend(self.running_score[i * threads : (i+1) * threads][dones[i * threads : (i+1) * threads]].tolist())
            self.running_score[dones] = 0

    def set_sp(self):
        self.ego_id = 0
        self.env_ego_id = 0
        # self.policy.set_sp()
        self.buffer = self.sp_buf

        # self.env.partners[0] = self.sp_partners
        # for partner in self.env.partners[0]:
        #     partner.player_id = 1 - self.ego_id

    def set_xp(self, ego_id):
        self.ego_id = ego_id
        self.envs.ego_ind = ego_id

        self.buffer = self.xp_buf0 if ego_id == 0 else self.xp_buf1

        # self.env.partners[0] = self.xp_partners
        # for partner in self.env.partners[0]:
        #     partner.player_id = 1 - self.ego_id

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

            # sp.txt
            # t: episode, Counter
            files.append(self.log_dir + "/sp.txt")

            # xp0.txt, xp1.txt
            # t: episode, Counter
            for i in range(len(self.agent_set)):
                files.append(self.log_dir + f"/xp_{i}_0.txt")
                files.append(self.log_dir + f"/xp_{i}_1.txt")

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

            average_score = np.mean(self.sp_scores)

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

            for i in range(len(self.agent_set)):
                avg0 = np.mean(self.xp_scores[0][i])
                avg1 = np.mean(self.xp_scores[1][i])
                general_log += f",avg_xp_{i}_0:{avg0},avg_xp_{i}_1:{avg1}"
                print(f"average xp score for {i} conv 0 is {avg0}.")
                print(f"average xp score for {i} conv 1 is {avg1}.")

            # self.log_train(train_infos, self.true_total_num_steps)
            print(train_infos)
            general_log += "," + ",".join(
                [f"{key}:{val}" for key, val in train_infos.items()]
            )

            files["log.txt"] = general_log

            files["sp.txt"] = get_histogram(self.sp_scores)
            print("Self-play Scores counts: ", files["sp.txt"])

            for i in range(len(self.agent_set)):
                for j in range(2):
                    files[f"xp_{i}_{j}.txt"] = get_histogram(self.xp_scores[j][i])
                    print(
                        f"Cross-play Scores counts (ego id {j}, convention {i}): ",
                        files[f"xp_{i}_{j}.txt"],
                    )

            for key, val in files.items():
                with open(f"{self.log_dir}/{key}", "a", encoding="UTF-8") as file:
                    file.write(f"episode:{episode},{val}\n")

    # def run(self):
    #     self.set_sp()
    #     self.setup_data()
    #     self.warmup()

    #     start = time.time()
    #     episodes = (
    #         int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
    #     )
    #     train_infos = None
    #     total_num_steps = 0

    #     self.best_i = None

    #     for episode in range(episodes):
    #         if self.use_linear_lr_decay:
    #             self.trainer.policy.lr_decay(episode, episodes)
    #         self.set_sp()
    #         self.collect_episode(turn_based=not self.all_args.simul_env)
    #         self.sp_scores = self.scores

    #         self.set_xp(0)
    #         self.collect_episode(turn_based=not self.all_args.simul_env)
    #         self.scores0 = self.scores

    #         self.set_xp(1)
    #         self.collect_episode(turn_based=not self.all_args.simul_env)
    #         self.scores1 = self.scores

    #         total_num_steps += self.episode_length
    #         # post process
    #         self.set_sp()
    #         self.log(train_infos, episode, episodes, total_num_steps, start)

    #         # compute return and update network
    #         self.compute_all()
    #         train_infos = self.train()
    #         print("DONE TRAINING:", episode)

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffers[0], self.buffers[1], self.buffers[2], len(self.agent_set))
        for buf in self.buffers:
            buf.reset_after_update()
        return train_infos

    def compute_all(self):
        self.set_sp()
        self.compute()

        self.set_xp(0)
        self.compute()

        self.set_xp(1)
        self.compute()
