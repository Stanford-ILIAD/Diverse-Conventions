# from MAPPO.utils.util import _t2n
# from .MCPolicy import MCPolicy
from .xd import XD
from MAPPO.utils.shared_buffer import SharedReplayBuffer
from MAPPO.main_player import MainPlayer

from MAPPO.rMAPPOPolicy import rMAPPOWrapper, R_MAPPOPolicy

from partner_agents import CentralizedMultiAgent, MixedAgent

import time
from collections import Counter

import numpy as np
import torch
import os


def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in sorted(Counter(x).items())])


class XDPlayer(MainPlayer):
    def __init__(
        self,
        config,
        policy: R_MAPPOPolicy,
        sp_buf: SharedReplayBuffer,
        xp_buf,
        agent_set,
        xp_weight,
        env_length,
    ):
        self._init_vars(config)

        # policy network
        self.policy = policy
        self.sp_buf = sp_buf
        self.xp_buf = xp_buf
        self.agent_set = agent_set
        self.xp_weight = xp_weight
        self.env_length = env_length

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = XD(
            self.all_args,
            self.policy,
            self.agent_set,
            self.xp_weight,
            self.all_args.use_average,
            device=self.device,
        )

        # buffer
        self.buffer = self.sp_buf

        self.envs.partners[0].clear()
        wrapped_agent_set = [rMAPPOWrapper(x, self.policy.critic) for i, x in enumerate(self.agent_set)]
        
        policies = [self.policy] + wrapped_agent_set
        self.envs.add_partner_agent(CentralizedMultiAgent(self, 1-self.ego_id, policies, self.all_args.n_rollout_threads))

    def collect_episode(self, buffer=None, length=None, save_scores=True): # REDO
        self.running_score = torch.zeros((self.envs.num_envs), device=self.device)

        if length is None:
            length = self.episode_length

        if save_scores:
            self.scores = [[] for _ in range(len(self.agent_set) + 1)]

        wrapped_agent_set = [rMAPPOWrapper(x, self.policy.critic) for i, x in enumerate(self.agent_set)]

        use_policies = [self.policy] + wrapped_agent_set

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

            for i in range(len(self.agent_set)):
                self.xp_buf[i].chooseinsert(self.turn_share_obs[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_obs[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_rnn_states[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_rnn_states_critic[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_actions[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_action_log_probs[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_values[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_rewards[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_masks[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_bad_masks[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_active_masks[threads * (i + 1) : threads * (i + 2)],
                                     self.turn_available_actions[threads * (i + 1) : threads * (i + 2)])
            

        self.sp_scores = self.scores[0]
        self.xp_scores = [self.scores[1 + i] for i in range(len(self.agent_set))]

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

    ## ALL GOOD BELOW

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

            # xp_i_0, xp_i_1, mp_i
            # t: episode, Counter
            for i in range(len(self.agent_set)):
                files.append(self.log_dir + f"/xp_{i}.txt")

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
                + f"avg_sp:{average_score}"
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
                avg0 = np.mean(self.xp_scores[i])
                general_log += f",avg_xp_{i}:{avg0}"
                print(f"average xp score for {i} is {avg0}.")

            train_infos["average_step_rewards"] = torch.mean(self.sp_buf.rewards).item()

            # self.log_train(train_infos, self.true_total_num_steps)
            print(train_infos)
            general_log += "," + ",".join(
                [f"{key}:{val}" for key, val in train_infos.items()]
            )

            files["log.txt"] = general_log

            files["sp.txt"] = get_histogram(self.sp_scores)
            print("Self-play Scores counts: ", files["sp.txt"])
            for i in range(len(self.agent_set)):
                files[f"xp_{i}.txt"] = get_histogram(self.xp_scores[i])
                print(
                    f"Cross-play Scores counts (convention {i}): ",
                    files[f"xp_{i}.txt"],
                )

            for key, val in files.items():
                with open(f"{self.log_dir}/{key}", "a", encoding="UTF-8") as file:
                    file.write(f"episode:{episode},{val}\n")

    def set_sp(self):
        self.buffer = self.sp_buf

    def set_xp(self, other_convention):
        self.buffer = self.xp_buf[other_convention]

    def compute(self):
        for i in range(len(self.agent_set)):
            self.set_xp(i)
            print("doing xp")
            self.compute_one()

        self.set_sp()
        self.compute_one()

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(
            self.sp_buf, self.xp_buf
        )
        for buf in [self.sp_buf] + self.xp_buf:
            buf.reset_after_update()
        return train_infos
