# from .utils.util import _t2n
from .rMAPPOPolicy import R_MAPPOPolicy as Policy
from .r_mappo import R_MAPPO as TrainAlgo
from .utils.shared_buffer import SharedReplayBuffer

import os
import time
from collections import Counter

import numpy as np
import torch


def get_histogram(x):
    return ",".join([f"{key}:{val}" for key, val in sorted(Counter(x).items())])


class MainPlayer:
    def __init__(self, config):
        self._init_vars(config)
        share_observation_space = self.envs.share_observation_space

        # policy network
        self.policy = Policy(
            self.all_args,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            device=self.device
        )

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space,
            share_observation_space,
            self.envs.action_space,
            self.device
        )

    def _init_vars(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.true_total_num_steps = 0
        self.ego_id = 0

    def collect_episode(self, buffer=None, length=None, save_scores=True):
        buffer = buffer or self.buffer
        self.running_score = torch.zeros((self.n_rollout_threads), device=self.device)
        if length is None:
            length = self.episode_length
        if save_scores:
            self.scores = []

        for _ in range(length):
            self.next_step(self.scores if save_scores else None)
            self.buffer.chooseinsert(self.turn_share_obs,
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
                                     self.turn_available_actions)

    def log(self, train_infos, episode, episodes, total_num_steps, start):
        # save model
        if episode % self.save_interval == 0 or episode == episodes - 1:
            self.save()

        if episode == 0:
            # Setup files
            files = []
            # log.txt
            # Env algo exp updates ... avg score, avg xp score
            print(self.log_dir)
            files.append(self.log_dir + "/log.txt")

            # sp.txt
            # t: episode, Counter
            files.append(self.log_dir + "/sp.txt")

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

            average_score = np.mean(self.scores)

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

            train_infos["average_step_rewards"] = torch.mean(self.buffer.rewards).item()

            # self.log_train(train_infos, self.true_total_num_steps)
            print(train_infos)
            general_log += "," + ",".join(
                [f"{key}:{val}" for key, val in train_infos.items()]
            )

            files["log.txt"] = general_log

            files["sp.txt"] = get_histogram(self.scores)
            print("Self-play Scores counts: ", files["sp.txt"])

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

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            print("Collecting Episode")
            self.collect_episode()
            print("Done collecting episode")
            total_num_steps += self.episode_length * self.n_rollout_threads

            self.log(train_infos, episode, episodes, total_num_steps, start)
            print("Done logging")
            self.compute()
            print("Done computing returns")
            train_infos = self.train()
            print("DONE TRAINING:", episode)

    @torch.no_grad()
    def next_step(self, scores=None):
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_state,
            rnn_state_critic
        ) = self.trainer.policy.get_actions(
            self.use_share_obs,
            self.use_obs,
            self.turn_rnn_states[:, self.ego_id],
            self.turn_rnn_states_critic[:, self.ego_id],
            self.turn_masks[0, self.ego_id],
            self.use_available_actions
        )

        self.turn_obs[:, self.ego_id] = self.use_obs
        self.turn_share_obs[:, self.ego_id] = self.use_share_obs
        self.turn_available_actions[:, self.ego_id] = self.use_available_actions
        self.turn_values[:, self.ego_id] = value
        self.turn_actions[:, self.ego_id] = action
        env_actions = action
        self.turn_action_log_probs[:, self.ego_id] = action_log_prob
        self.turn_rnn_states[:, self.ego_id] = rnn_state
        self.turn_rnn_states_critic[:, self.ego_id] = rnn_state_critic
        self.turn_active_masks[:, 1 - self.ego_id] = 0
        self.turn_active_masks[:, self.ego_id] = 1
        vobs, rewards, done, info = self.envs.step(
            env_actions
        )
        dones = done.to(torch.bool)
        rewards = rewards
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
            scores.extend(self.running_score[dones].tolist())
            self.running_score[dones] = 0

    def setup_data(self):
        self.turn_obs = torch.zeros((self.n_rollout_threads,*self.buffer.obs.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_share_obs = torch.zeros((self.n_rollout_threads,*self.buffer.share_obs.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_available_actions = torch.zeros((self.n_rollout_threads,*self.buffer.available_actions.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_values = torch.zeros((self.n_rollout_threads,*self.buffer.value_preds.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_actions = torch.zeros((self.n_rollout_threads,*self.buffer.actions.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_action_log_probs = torch.zeros((self.n_rollout_threads,*self.buffer.action_log_probs.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_rnn_states = torch.zeros((self.n_rollout_threads,*self.buffer.rnn_states.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_rnn_states_critic = torch.zeros_like(self.turn_rnn_states)
        self.turn_masks = torch.ones((self.n_rollout_threads,*self.buffer.masks.shape[2:]), dtype=torch.float, device=self.device)
        self.turn_active_masks = torch.ones_like(self.turn_masks)
        self.turn_bad_masks = torch.ones_like(self.turn_masks)
        self.turn_rewards = torch.zeros((self.n_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=torch.float, device=self.device)

        self.turn_rewards_since_last_action = torch.zeros_like(self.turn_rewards)

    def warmup(self):
        # reset env
        vobs = self.envs.reset()

        # replay buffer
        self.use_obs = vobs.obs.clone()
        self.use_share_obs = vobs.state.clone()
        self.use_available_actions = vobs.action_mask.clone()
        # self.use_is_active = _t2n(vobs.active).copy()
        self.running_score = torch.zeros((self.n_rollout_threads), dtype=torch.float, device=self.device)


    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        # print(self.buffer.share_obs[-1].flatten(0, 1).shape)
        next_values = self.trainer.policy.get_values(
            self.buffer.share_obs[-1].flatten(0, 1),
            self.buffer.rnn_states_critic[-1].flatten(0, 1),
            self.buffer.masks[-1].flatten(0, 1)
        )
        # next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        next_values = next_values.unflatten(0, (self.n_rollout_threads, -1))  # torch.split(next_values, next_values.size(dim=0)//self.n_rollout_threads)
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.reset_after_update()  # TODO: Change this?
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        print("SAVED TO", self.save_dir)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
