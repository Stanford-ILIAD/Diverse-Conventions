from MAPPO.utils.util import _t2n
from .MCPolicy import MCPolicy
from .xd import XD
from MAPPO.utils.shared_buffer import SharedReplayBuffer
from MAPPO.main_player import MainPlayer

from partner_agents import CentralizedAgent, MixedAgent

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
        policy: MCPolicy,
        sp_buf: SharedReplayBuffer,
        xp_buf0,
        xp_buf1,
        mp_buf,
        agent_set,
        xp_weight,
        mp_weight,
        mix_prob,
        env_length,
    ):
        self._init_vars(config)

        # policy network
        self.policy = policy
        self.sp_buf = sp_buf
        self.xp_buf0 = xp_buf0
        self.xp_buf1 = xp_buf1
        self.mp_buf = mp_buf
        self.agent_set = agent_set
        self.xp_weight = xp_weight
        self.mp_weight = mp_weight
        self.mix_prob = mix_prob
        self.env_length = env_length

        self.best_i = None

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = XD(
            self.all_args,
            self.policy,
            self.agent_set,
            self.xp_weight,
            self.mp_weight,
            self.all_args.use_average,
            device=self.device,
        )

        # buffer
        self.buffer = self.sp_buf

    def set_sp(self):
        self.ego_id = 0
        self.env.ego_ind = 0
        self.policy.set_sp()
        self.buffer = self.sp_buf

        partner = CentralizedAgent(self, 1 - self.ego_id)

        self.env.partners[0].clear()
        self.env.add_partner_agent(partner)

    def set_xp(self, ego_id, other_convention):
        self.ego_id = ego_id
        self.env.ego_ind = ego_id
        self.policy.set_xp(ego_id, other_convention)

        if ego_id == 0:
            self.buffer = self.xp_buf0[other_convention]
        else:
            self.buffer = self.xp_buf1[other_convention]

        partner = CentralizedAgent(
            self, 1 - self.ego_id, self.agent_set[other_convention]
        )

        self.env.partners[0].clear()
        self.env.add_partner_agent(partner, 1 - self.ego_id)

    def set_mp(self, other_convention):
        self.ego_id = 0
        self.env.ego_ind = 0
        self.policy.set_mp()
        self.buffer = self.mp_buf[other_convention]

        partner = MixedAgent(
            self, 1 - self.ego_id, self.agent_set[other_convention], self.mix_prob
        )

        self.env.partners[0].clear()
        self.env.add_partner_agent(partner)

        self.mp_ind = other_convention

    def collect_mp_episode(self, turn_based=True):
        partner = self.env.partners[0][0]
        self.scores = []
        self.running_score = 0
        step = 0
        while step < self.episode_length:
            (
                self.use_obs,
                self.use_share_obs,
                self.use_available_actions,
            ) = self.env.reset()

            phase1len = np.random.randint(self.env_length - 1) + 1
            partner.second_phase = False
            for _ in range(phase1len):
                done = self.next_mp_step(self.scores, partner)
                if done:
                    break
            if done:
                continue
            partner.second_phase = True
            while not done and step < self.episode_length:
                done = self.next_step(self.scores)

                # insert turn data into buffer
                if turn_based:
                    self.buffer.chooseinsert(
                        self.turn_share_obs,
                        self.turn_obs,
                        self.turn_rnn_states,
                        self.turn_rnn_states_critic,
                        self.turn_actions,
                        self.turn_action_log_probs,
                        self.turn_values,
                        self.turn_rewards,
                        self.turn_masks,
                        self.turn_bad_masks,
                        self.turn_active_masks,
                        self.turn_available_actions,
                    )
                else:
                    self.buffer.insert(
                        self.turn_share_obs,
                        self.turn_obs,
                        self.turn_rnn_states,
                        self.turn_rnn_states_critic,
                        self.turn_actions,
                        self.turn_action_log_probs,
                        self.turn_values,
                        self.turn_rewards,
                        self.turn_masks,
                        self.turn_bad_masks,
                        self.turn_active_masks,
                        self.turn_available_actions,
                    )
                step += 1

    def next_mp_step(self, scores, partner):
        self.trainer.prep_rollout()

        actor = (
            partner.other_policy
            if np.random.rand() > self.mix_prob
            else self.trainer.policy.actor
        )
        critic = self.trainer.policy.critic

        (action, action_log_prob, rnn_state) = actor(
            self.use_obs,
            self.turn_rnn_states[0, self.ego_id],
            self.turn_masks[0, self.ego_id],
            self.use_available_actions,
        )
        (value, rnn_state_critic) = critic(
            self.use_share_obs,
            self.turn_rnn_states_critic[0, self.ego_id],
            self.turn_masks[0, self.ego_id],
        )

        self.turn_obs[0, self.ego_id] = self.use_obs.copy()
        self.turn_share_obs[0, self.ego_id] = self.use_share_obs.copy()
        self.turn_available_actions[0, self.ego_id] = self.use_available_actions.copy()
        self.turn_values[0, self.ego_id] = _t2n(value)
        self.turn_actions[0, self.ego_id] = _t2n(action)
        env_actions = _t2n(action)
        self.turn_action_log_probs[0, self.ego_id] = _t2n(action_log_prob)
        self.turn_rnn_states[0, self.ego_id] = _t2n(rnn_state)
        self.turn_rnn_states_critic[0, self.ego_id] = _t2n(rnn_state_critic)
        self.turn_active_masks[0, 1 - self.ego_id] = 0
        self.turn_active_masks[0, self.ego_id] = 1
        (obs, share_obs, available_actions), rewards, done, info = self.env.step(
            env_actions
        )
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()
        self.turn_rewards[0, self.ego_id] = rewards

        self.running_score += rewards

        if done:
            self.turn_masks[0, self.ego_id] = 0
            self.turn_rnn_states[0, self.ego_id] = 0
            self.turn_rnn_states_critic[0, self.ego_id] = 0

            if scores is not None:
                scores.append(self.running_score)

            self.running_score = 0
        else:
            self.turn_masks[0, self.ego_id] = 1

        return done

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
                files.append(self.log_dir + f"/xp_{i}_0.txt")
                files.append(self.log_dir + f"/xp_{i}_1.txt")
                files.append(self.log_dir + f"/mp_{i}.txt")

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
                avg0 = np.mean(self.xp_scores[0][i])
                avg1 = np.mean(self.xp_scores[1][i])
                general_log += f",avg_xp_{i}_0:{avg0},avg_xp_{i}_1:{avg1}"
                print(f"average xp score for {i} conv 0 is {avg0}.")
                print(f"average xp score for {i} conv 1 is {avg1}.")

            train_infos["average_step_rewards"] = np.mean(self.sp_buf.rewards)

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

            for i in range(len(self.agent_set)):
                files[f"mp_{i}.txt"] = get_histogram(self.mp_scores[i])
                print(
                    f"Mix-play Scores counts (convention {i}): ", files[f"mp_{i}.txt"]
                )
            for key, val in files.items():
                with open(f"{self.log_dir}/{key}", "a", encoding="UTF-8") as file:
                    file.write(f"episode:{episode},{val}\n")

    def run(self):
        self.set_sp()
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
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self.set_sp()
            self.collect_episode(turn_based=not self.all_args.simul_env)
            self.sp_scores = self.scores

            self.xp_scores = [([[]] * len(self.agent_set)) for _ in range(2)]
            self.mp_scores = [[]] * len(self.agent_set)

            agent_iterator = range(len(self.agent_set))
            
            if len(self.agent_set) > 0 and episode % len(self.agent_set) == 0:
                self.best_i = None

            if self.best_i is not None:
                agent_iterator = [self.best_i]
            
            for i in agent_iterator:
                if self.xp_weight != 0:
                    self.set_xp(0, i)
                    self.collect_episode(turn_based=not self.all_args.simul_env)
                    self.xp_scores[0][i] = self.scores

                    self.set_xp(1, i)
                    self.collect_episode(turn_based=not self.all_args.simul_env)
                    self.xp_scores[1][i] = self.scores

                if self.mp_weight != 0:
                    self.set_mp(i)
                    self.collect_mp_episode(turn_based=not self.all_args.simul_env)
                    self.mp_scores[i] = self.scores
            total_num_steps += self.episode_length
            # post process
            self.set_sp()
            self.log(train_infos, episode, episodes, total_num_steps, start)

            # compute return and update network
            self.compute_all()
            train_infos = self.train()
            print("DONE TRAINING:", episode)

    def compute_all(self):
        for i in range(len(self.agent_set)):
            self.set_xp(0, i)
            self.compute()

            self.set_xp(1, i)
            self.compute()

            self.set_mp(i)
            self.compute()

        self.set_sp()
        self.compute()

    def train(self):
        """Train policies with data in buffer."""
        self.trainer.prep_training()
        train_infos = self.trainer.train(
            self.sp_buf, self.xp_buf0, self.xp_buf1, self.mp_buf, best_i=self.best_i
        )
        for buf in [self.sp_buf] + self.xp_buf0 + self.xp_buf1 + self.mp_buf:
            buf.reset_after_update()
        self.best_i = train_infos["best_i"]
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        print(os.path.exists(self.save_dir))
        os.makedirs(self.save_dir, exist_ok=True)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        print("SAVED TO", self.save_dir)
        policy_critic_sp = self.trainer.policy.sp_critic
        torch.save(policy_critic_sp.state_dict(), str(self.save_dir) + "/sp_critic.pt")
        for i in range(len(self.agent_set)):
            policy_critic_0 = self.trainer.policy.xp_critic0[i]
            torch.save(
                policy_critic_0.state_dict(),
                str(self.save_dir) + "/xp_critic0_" + str(i) + ".pt",
            )
            policy_critic_1 = self.trainer.policy.xp_critic1[i]
            torch.save(
                policy_critic_1.state_dict(),
                str(self.save_dir) + "/xp_critic1_" + str(i) + ".pt",
            )

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor.pt")
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_sp_state_dict = torch.load(
                str(self.model_dir) + "/sp_critic.pt"
            )
            self.policy.sp_critic.load_state_dict(policy_critic_sp_state_dict)

            for i in range(len(self.agent_set)):
                xp_crit_0 = torch.load(
                    str(self.model_dir) + "/xp_critic0_" + str(i) + ".pt"
                )
                self.policy.xp_critic0[i].load_state_dict(xp_crit_0)

                xp_crit_1 = torch.load(
                    str(self.model_dir) + "/xp_critic1_" + str(i) + ".pt"
                )
                self.policy.xp_critic1[i].load_state_dict(xp_crit_1)
