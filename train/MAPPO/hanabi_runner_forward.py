from .rMAPPOPolicy import R_MAPPOPolicy as Policy
from .r_mappo import R_MAPPO as TrainAlgo

import os
import time
from collections import Counter

import numpy as np
import torch


from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()

class HanabiRunner:
    """Runner class to perform training, evaluation. and data collection for Hanabi. See parent class for details."""
    def __init__(self, config):
        self._init_vars(config)
        self.true_total_num_steps = 0
        self.ego_id = 0

    def _init_vars(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
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
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0])

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

            train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)

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
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        train_infos = None
        total_num_steps = 0
        # TODO: initialize train_infos and total_num_steps?

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self.collect_episode()
            total_num_steps += self.episode_length * self.n_rollout_threads

            self.log(train_infos, episode, episodes, total_num_steps, start)

            self.compute()
            train_infos = self.train()
            print("DONE TRAINING:", episode)
            

            # self.scores = []
            # for step in range(self.episode_length):
            #     self.reset_choose = np.zeros(self.n_rollout_threads) == 1.0
            #     # Sample actions
            #     self.collect(step)

            #     if step == 0 and episode > 0:
            #         # deal with the data of the last index in buffer
            #         self.buffer.share_obs[-1] = self.turn_share_obs.copy()
            #         self.buffer.obs[-1] = self.turn_obs.copy()
            #         self.buffer.available_actions[-1] = self.turn_available_actions.copy()
            #         self.buffer.active_masks[-1] = self.turn_active_masks.copy()

            #         # deal with rewards
            #         # 1. shift all rewards
            #         self.buffer.rewards[0:self.episode_length-1] = self.buffer.rewards[1:]
            #         # 2. last step rewards
            #         self.buffer.rewards[-1] = self.turn_rewards.copy()

            #         # compute return and update network
            #         self.compute()
            #         train_infos = self.train()

            #     # insert turn data into buffer
            #     self.buffer.chooseinsert(self.turn_share_obs,
            #                             self.turn_obs,
            #                             self.turn_rnn_states,
            #                             self.turn_rnn_states_critic,
            #                             self.turn_actions,
            #                             self.turn_action_log_probs,
            #                             self.turn_values,
            #                             self.turn_rewards,
            #                             self.turn_masks,
            #                             self.turn_bad_masks,
            #                             self.turn_active_masks,
            #                             self.turn_available_actions)
            #     # env reset
            #     # obs, share_obs, available_actions = self.envs.reset(self.reset_choose)
            #     # share_obs = share_obs if self.use_centralized_V else obs

            #     # self.use_obs[self.reset_choose] = obs[self.reset_choose]
            #     # self.use_share_obs[self.reset_choose] = share_obs[self.reset_choose]
            #     # self.use_available_actions[self.reset_choose] = available_actions[self.reset_choose]

            # # post process
            # total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # # save model
            # if (episode % self.save_interval == 0 or episode == episodes - 1):
            #     self.save()

            # # log information
            # if episode % self.log_interval == 0 and episode > 0:
            #     end = time.time()
            #     print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
            #             .format(self.all_args.hanabi_name,
            #                     self.algorithm_name,
            #                     self.experiment_name,
            #                     episode,
            #                     episodes,
            #                     total_num_steps,
            #                     self.num_env_steps,
            #                     int(total_num_steps / (end - start))))

            #     if self.env_name == "Hanabi":
            #         average_score = np.mean(self.scores) if len(self.scores) > 0 else 0.0
            #         print("average score is {}.".format(average_score))
            #         print("Scores counts:", sorted(Counter(self.scores).items()))
            #         self.writter.add_scalars('average_score', {'average_score': average_score}, self.true_total_num_steps)

            #     train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)

            #     self.log_train(train_infos, self.true_total_num_steps)

            # # eval
            # if episode % self.eval_interval == 0 and self.use_eval:
            #     self.eval(self.true_total_num_steps)

    # @torch.no_grad()
    # def collect(self, step):
    #     for current_agent_id in range(self.num_agents):
    #         env_actions = np.ones((self.n_rollout_threads, *self.buffer.actions.shape[3:]), dtype=np.float32)*(-1.0)
    #         choose = np.any(self.use_available_actions == 1, axis=1)
    #         if ~np.any(choose):
    #             self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
    #             break

    #         self.trainer.prep_rollout()
    #         value, action, action_log_prob, rnn_state, rnn_state_critic \
    #             = self.trainer.policy.get_actions(self.use_share_obs[choose],
    #                                             self.use_obs[choose],
    #                                             self.turn_rnn_states[choose, current_agent_id],
    #                                             self.turn_rnn_states_critic[choose, current_agent_id],
    #                                             self.turn_masks[choose, current_agent_id],
    #                                             self.use_available_actions[choose])

    #         self.turn_obs[choose, current_agent_id] = self.use_obs[choose].copy()
    #         self.turn_share_obs[choose, current_agent_id] = self.use_share_obs[choose].copy()
    #         self.turn_available_actions[choose, current_agent_id] = self.use_available_actions[choose].copy()
    #         self.turn_values[choose, current_agent_id] = _t2n(value)
    #         self.turn_actions[choose, current_agent_id] = _t2n(action)
    #         env_actions[choose] = _t2n(action)
    #         self.turn_action_log_probs[choose, current_agent_id] = _t2n(action_log_prob)
    #         self.turn_rnn_states[choose, current_agent_id] = _t2n(rnn_state)
    #         self.turn_rnn_states_critic[choose, current_agent_id] = _t2n(rnn_state_critic)

    #         obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions)

    #         self.true_total_num_steps += (choose==True).sum()
    #         share_obs = share_obs if self.use_centralized_V else obs

    #         # truly used value
    #         self.use_obs = obs.copy()
    #         self.use_share_obs = share_obs.copy()
    #         self.use_available_actions = available_actions.copy()

    #         # rearrange reward
    #         # reward of step 0 will be thrown away.
    #         self.turn_rewards[choose, current_agent_id] = self.turn_rewards_since_last_action[choose, current_agent_id].copy()
    #         self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0
    #         self.turn_rewards_since_last_action[choose] += rewards[choose]

    #         # done==True env

    #         # deal with reset_choose
    #         self.reset_choose[dones == True] = np.ones((dones == True).sum(), dtype=bool)

    #         # deal with all agents
    #         self.use_available_actions[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)
    #         self.turn_masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)
    #         self.turn_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
    #         self.turn_rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

    #         # deal with the current agent
    #         self.turn_active_masks[dones == True, current_agent_id] = np.ones(((dones == True).sum(), 1), dtype=np.float32)

    #         # deal with the left agents
    #         left_agent_id = current_agent_id + 1
    #         left_agents_num = self.num_agents - left_agent_id
    #         self.turn_active_masks[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)

    #         self.turn_rewards[dones == True, left_agent_id:] = self.turn_rewards_since_last_action[dones == True, left_agent_id:]
    #         self.turn_rewards_since_last_action[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)

    #         # other variables use what at last time, action will be useless.
    #         self.turn_values[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
    #         self.turn_obs[dones == True, left_agent_id:] = 0
    #         self.turn_share_obs[dones == True, left_agent_id:] = 0

    #         # done==False env
    #         # deal with current agent
    #         self.turn_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)
    #         self.turn_active_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)

    #         # done==None
    #         # pass

    #         for done, info in zip(dones, infos):
    #             if done:
    #                 if 'score' in info.keys():
    #                     self.scores.append(info['score'])

    def collect_episode(self, buffer=None, length=None, save_scores=True):
        buffer = buffer or self.buffer
        self.use_obs, self.use_share_obs, self.use_available_actions = self.env.reset() # TODO: Update for vectorized
        self.running_score = np.zeros((self.n_rollout_threads), dtype=np.float32)

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

    @torch.no_grad()
    def next_step(self, scores=None):
        current_agent_id = self.ego_id
        env_actions = np.ones((self.n_rollout_threads, *self.buffer.actions.shape[3:]), dtype=np.float32)*(-1.0)
        choose = np.any(self.use_available_actions == 1, axis=1)
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer.policy.get_actions(self.use_share_obs[choose],
                                                self.use_obs[choose],
                                                self.turn_rnn_states[choose, current_agent_id],
                                                self.turn_rnn_states_critic[choose, current_agent_id],
                                                self.turn_masks[choose, current_agent_id],
                                                self.use_available_actions[choose])
        self.turn_obs[choose, current_agent_id] = self.use_obs[choose].copy()
        self.turn_share_obs[choose, current_agent_id] = self.use_share_obs[choose].copy()
        self.turn_available_actions[choose, current_agent_id] = self.use_available_actions[choose].copy()
        self.turn_values[choose, current_agent_id] = _t2n(value)
        self.turn_actions[choose, current_agent_id] = _t2n(action)
        env_actions[choose] = _t2n(action)
        self.turn_action_log_probs[choose, current_agent_id] = _t2n(action_log_prob)
        self.turn_rnn_states[choose, current_agent_id] = _t2n(rnn_state)
        self.turn_rnn_states_critic[choose, current_agent_id] = _t2n(rnn_state_critic)

        obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(env_actions)

        self.true_total_num_steps += (choose==True).sum()

        # truly used value
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()

        # rearrange reward
        # reward of step 0 will be thrown away.
        self.turn_rewards[choose, current_agent_id] = self.turn_rewards_since_last_action[choose, current_agent_id].copy()
        self.turn_rewards_since_last_action[choose, current_agent_id] = 0.0
        self.turn_rewards_since_last_action[choose] += rewards[choose]

        # done==True env

        # deal with reset_choose
        self.reset_choose[dones == True] = np.ones((dones == True).sum(), dtype=bool)

        # deal with all agents
        self.use_available_actions[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)
        self.turn_masks[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, 1), dtype=np.float32)
        self.turn_rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        self.turn_rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        # deal with the current agent
        self.turn_active_masks[dones == True, current_agent_id] = np.ones(((dones == True).sum(), 1), dtype=np.float32)

        # deal with the left agents
        left_agent_id = current_agent_id + 1
        left_agents_num = self.num_agents - left_agent_id
        self.turn_active_masks[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)

        self.turn_rewards[dones == True, left_agent_id:] = self.turn_rewards_since_last_action[dones == True, left_agent_id:]
        self.turn_rewards_since_last_action[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)

        # other variables use what at last time, action will be useless.
        self.turn_values[dones == True, left_agent_id:] = np.zeros(((dones == True).sum(), left_agents_num, 1), dtype=np.float32)
        self.turn_obs[dones == True, left_agent_id:] = 0
        self.turn_share_obs[dones == True, left_agent_id:] = 0

        # done==False env
        # deal with current agent
        self.turn_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)
        self.turn_active_masks[dones == False, current_agent_id] = np.ones(((dones == False).sum(), 1), dtype=np.float32)

        # TODO: Running scores?
        self.running_score[choose] += rewards[choose]
        if scores is not None and np.any(dones == True):
            scores.append(self.running_score[dones == True].tolist())
            self.running_scores[dones == True] = 0

        # done==None
        # pass

        # for done, info in zip(dones, infos):
        #     if done:
        #         if 'score' in info.keys():
        #             self.scores.append(info['score'])

    # @torch.no_grad()
    # def eval(self, total_num_steps):
    #     eval_envs = self.eval_envs

    #     eval_scores = []

    #     eval_finish = False
    #     eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0

    #     eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)

    #     eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
    #     eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

    #     while True:
    #         if eval_finish:
    #             break
    #         for agent_id in range(self.num_agents):
    #             eval_actions = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32) * (-1.0)
    #             eval_choose = np.any(eval_available_actions == 1, axis=1)

    #             if ~np.any(eval_choose):
    #                 eval_finish = True
    #                 break

    #             self.trainer.prep_rollout()
    #             eval_action, eval_rnn_state = self.trainer.policy.act(eval_obs[eval_choose],
    #                                                             eval_rnn_states[eval_choose, agent_id],
    #                                                             eval_masks[eval_choose, agent_id],
    #                                                             eval_available_actions[eval_choose],
    #                                                             deterministic=True)

    #             eval_actions[eval_choose] = _t2n(eval_action)
    #             eval_rnn_states[eval_choose, agent_id] = _t2n(eval_rnn_state)

    #             # Obser reward and next obs
    #             eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(eval_actions)

    #             eval_available_actions[eval_dones == True] = np.zeros(((eval_dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)

    #             for eval_done, eval_info in zip(eval_dones, eval_infos):
    #                 if eval_done:
    #                     if 'score' in eval_info.keys():
    #                         eval_scores.append(eval_info['score'])

    #     eval_average_score = np.mean(eval_scores)
    #     print("eval average score is {}.".format(eval_average_score))
    #     self.writter.add_scalars('eval_average_score', {'eval_average_score': eval_average_score}, total_num_steps)


    # @torch.no_grad()
    # def eval_100k(self, eval_games=100000):
    #     eval_envs = self.eval_envs
    #     trials = int(eval_games/self.n_eval_rollout_threads)

    #     eval_scores = []
    #     for trial in range(trials):
    #         print("trail is {}".format(trial))
    #         eval_finish = False
    #         eval_reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0

    #         eval_obs, eval_share_obs, eval_available_actions = eval_envs.reset(eval_reset_choose)

    #         eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
    #         eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

    #         while True:
    #             if eval_finish:
    #                 break
    #             for agent_id in range(self.num_agents):
    #                 eval_actions = np.ones((self.n_eval_rollout_threads, 1), dtype=np.float32) * (-1.0)
    #                 eval_choose = np.any(eval_available_actions == 1, axis=1)

    #                 if ~np.any(eval_choose):
    #                     eval_finish = True
    #                     break

    #                 self.trainer.prep_rollout()
    #                 eval_action, eval_rnn_state = self.trainer.policy.act(eval_obs[eval_choose],
    #                                                                 eval_rnn_states[eval_choose, agent_id],
    #                                                                 eval_masks[eval_choose, agent_id],
    #                                                                 eval_available_actions[eval_choose],
    #                                                                 deterministic=True)

    #                 eval_actions[eval_choose] = _t2n(eval_action)
    #                 eval_rnn_states[eval_choose, agent_id] = _t2n(eval_rnn_state)

    #                 # Obser reward and next obs
    #                 eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = eval_envs.step(eval_actions)

    #                 eval_available_actions[eval_dones == True] = np.zeros(((eval_dones == True).sum(), *self.buffer.available_actions.shape[3:]), dtype=np.float32)

    #                 for eval_done, eval_info in zip(eval_dones, eval_infos):
    #                     if eval_done:
    #                         if 'score' in eval_info.keys():
    #                             eval_scores.append(eval_info['score'])

    #     eval_average_score = np.mean(eval_scores)
    #     print("eval average score is {}.".format(eval_average_score))

    def setup_data(self):
        self.turn_obs = np.zeros((self.n_rollout_threads,*self.buffer.obs.shape[2:]), dtype=np.float32)
        self.turn_share_obs = np.zeros((self.n_rollout_threads,*self.buffer.share_obs.shape[2:]), dtype=np.float32)
        self.turn_available_actions = np.zeros((self.n_rollout_threads,*self.buffer.available_actions.shape[2:]), dtype=np.float32)
        self.turn_values = np.zeros((self.n_rollout_threads,*self.buffer.value_preds.shape[2:]), dtype=np.float32)
        self.turn_actions = np.zeros((self.n_rollout_threads,*self.buffer.actions.shape[2:]), dtype=np.float32)
        self.turn_action_log_probs = np.zeros((self.n_rollout_threads,*self.buffer.action_log_probs.shape[2:]), dtype=np.float32)
        self.turn_rnn_states = np.zeros((self.n_rollout_threads,*self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        self.turn_rnn_states_critic = np.zeros_like(self.turn_rnn_states)
        self.turn_masks = np.ones((self.n_rollout_threads,*self.buffer.masks.shape[2:]), dtype=np.float32)
        self.turn_active_masks = np.ones_like(self.turn_masks)
        self.turn_bad_masks = np.ones_like(self.turn_masks)
        self.turn_rewards = np.zeros((self.n_rollout_threads, *self.buffer.rewards.shape[2:]), dtype=np.float32)

        self.turn_rewards_since_last_action = np.zeros_like(self.turn_rewards)

    def warmup(self):
        # reset env
        # TODO: use Pantheon interface instead
        self.reset_choose = np.ones(self.n_rollout_threads) == 1.0
        obs, share_obs, available_actions = self.envs.reset(self.reset_choose)

        share_obs = share_obs if self.use_centralized_V else obs

        # replay buffer
        self.use_obs = obs.copy()
        self.use_share_obs = share_obs.copy()
        self.use_available_actions = available_actions.copy()
        # TODO: running_score?

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.chooseafter_update()  # TODO: Change this?
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
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
 
    # def log_train(self, train_infos, total_num_steps):
    #     """
    #     Log training info.
    #     :param train_infos: (dict) information about training update.
    #     :param total_num_steps: (int) total number of training env steps.
    #     """
    #     for k, v in train_infos.items():
    #         self.writter.add_scalars(k, {k: v}, total_num_steps)

    # def log_env(self, env_infos, total_num_steps):
    #     """
    #     Log env info.
    #     :param env_infos: (dict) information about env state.
    #     :param total_num_steps: (int) total number of training env steps.
    #     """
    #     for k, v in env_infos.items():
    #         if len(v)>0:
    #             self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

