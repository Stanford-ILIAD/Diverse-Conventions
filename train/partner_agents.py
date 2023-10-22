# from pantheonrl.common.agents import Agent

# from pantheonrl.algos.bc import reconstruct_policy
# from pantheonrl.common.agents import StaticPolicyAgent

from pantheonrl_extension.vectoragent import VectorAgent

# from MAPPO.utils.util import _t2n
from MAPPO.main_player import MainPlayer
# from MAPPO.r_actor_critic import R_Actor

# import numpy as np
import torch
import torch.nn.functional as F

# from pantheonrl.common.util import action_from_policy


class CentralizedAgent(VectorAgent):
    def __init__(self, cent_player: MainPlayer, player_id: int, policy=None):
        self.cent_player = cent_player
        self.player_id = player_id
        if policy is None:
            self.actor = self.cent_player.trainer.policy.actor
        else:
            self.actor = policy

    def get_action(self, obs, record=True):
        available_actions = obs.action_mask
        share_obs = obs.state
        # choose = obs.active
        obs = obs.obs

        self.cent_player.trainer.prep_rollout()

        (action, action_log_prob, rnn_state) = self.actor(
            obs,
            self.cent_player.turn_rnn_states[:, self.player_id],
            self.cent_player.turn_masks[:, self.player_id],
            available_actions,
        )
        critic = self.cent_player.trainer.policy.critic
        (value, rnn_state_critic) = critic(
            share_obs,
            self.cent_player.turn_rnn_states_critic[:, self.player_id],
            self.cent_player.turn_masks[:, self.player_id],
        )

        if record:
            self.cent_player.turn_obs[:, self.player_id] = obs
            self.cent_player.turn_share_obs[:, self.player_id] = share_obs
            self.cent_player.turn_available_actions[
                :, self.player_id
            ] = available_actions
            self.cent_player.turn_values[:, self.player_id] = value
            self.cent_player.turn_actions[:, self.player_id] = action
            self.cent_player.turn_action_log_probs[:, self.player_id] = action_log_prob
            self.cent_player.turn_rnn_states[:, self.player_id] = rnn_state
            self.cent_player.turn_rnn_states_critic[:, self.player_id] = rnn_state_critic
            self.cent_player.turn_rewards[:, self.player_id] = 0
            self.cent_player.turn_active_masks[:, self.player_id] = 1

        return action

    def update(self, rewards, dones):
        rewards = rewards
        dones = dones.to(torch.bool)
        # print(dones)
        # print(rewards, self.cent_player.turn_rewards[:, 1])
        self.cent_player.turn_rewards[:, self.player_id] += rewards[:, None]

        self.cent_player.turn_masks[dones, self.player_id] = 0
        self.cent_player.turn_rnn_states[dones, self.player_id] = 0
        self.cent_player.turn_rnn_states_critic[dones, self.player_id] = 0

        self.cent_player.turn_masks[~dones, self.player_id] = 1


class CentralizedMultiAgent(VectorAgent):
    def __init__(self, cent_player: MainPlayer, player_id: int, policies, threads):
        self.cent_player = cent_player
        self.player_id = player_id
        self.policies = policies

        self.threads = threads

    def get_action(self, obs, record=True):
        available_actions = obs.action_mask
        share_obs = obs.state
        # choose = obs.active
        obs = obs.obs

        threads = self.threads

        self.cent_player.trainer.prep_rollout()

        for i, p in enumerate(self.policies):
            (
                self.cent_player.turn_values[i * threads : (i + 1) * threads, self.player_id],
                self.cent_player.turn_actions[i * threads : (i + 1) * threads, self.player_id],
                self.cent_player.turn_action_log_probs[i * threads : (i + 1) * threads, self.player_id],
                self.cent_player.turn_rnn_states[i * threads : (i + 1) * threads, self.player_id],
                self.cent_player.turn_rnn_states_critic[i * threads : (i + 1) * threads, self.player_id]
            ) = p.get_actions(
                share_obs[i * threads : (i + 1) * threads],
                obs[i * threads : (i + 1) * threads],
                self.cent_player.turn_rnn_states[i * threads : (i + 1) * threads, self.player_id],
                self.cent_player.turn_rnn_states_critic[i * threads : (i + 1) * threads, self.player_id],
                self.cent_player.turn_masks[i * threads : (i + 1) * threads, self.player_id],
                available_actions[i * threads : (i + 1) * threads]
            )

        # (action, action_log_prob, rnn_state) = self.actor(
        #     obs,
        #     self.cent_player.turn_rnn_states[:, self.player_id],
        #     self.cent_player.turn_masks[:, self.player_id],
        #     available_actions,
        # )
        # critic = self.cent_player.trainer.policy.critic
        # (value, rnn_state_critic) = critic(
        #     share_obs,
        #     self.cent_player.turn_rnn_states_critic[:, self.player_id],
        #     self.cent_player.turn_masks[:, self.player_id],
        # )

        self.cent_player.turn_obs[:, self.player_id] = obs
        self.cent_player.turn_share_obs[:, self.player_id] = share_obs
        self.cent_player.turn_available_actions[:, self.player_id] = available_actions
        # self.cent_player.turn_values[:, self.player_id] = value
        # self.cent_player.turn_actions[:, self.player_id] = action
        # self.cent_player.turn_action_log_probs[:, self.player_id] = action_log_prob
        # self.cent_player.turn_rnn_states[:, self.player_id] = rnn_state
        # self.cent_player.turn_rnn_states_critic[:, self.player_id] = rnn_state_critic
        self.cent_player.turn_rewards[:, self.player_id] = 0
        self.cent_player.turn_active_masks[:, self.player_id] = 1

        return self.cent_player.turn_actions[:, self.player_id]

    def update(self, rewards, dones):
        rewards = rewards
        dones = dones.to(torch.bool)
        # print(dones)
        # print(rewards, self.cent_player.turn_rewards[:, 1])
        self.cent_player.turn_rewards[:, self.player_id] += rewards[:, None]

        self.cent_player.turn_masks[dones, self.player_id] = 0
        self.cent_player.turn_rnn_states[dones, self.player_id] = 0
        self.cent_player.turn_rnn_states_critic[dones, self.player_id] = 0

        self.cent_player.turn_masks[~dones, self.player_id] = 1


class MixedAgent(VectorAgent):
    def __init__(self, cent_player: MainPlayer, player_id: int, mix_policies, length):
        self.cent_player = cent_player
        self.player_id = player_id
        self.mix_policies = mix_policies

        self.step = 0
        self.length = length

    def get_action(self, obs, record=True):
        available_actions = obs.action_mask
        share_obs = obs.state
        # choose = obs.active
        obs = obs.obs

        self.cent_player.trainer.prep_rollout()

        mix_mask = (torch.rand((self.length - 1), device=self.cent_player.device) < 0.5)

        if self.step >= self.length:
            mix_mask[:self.step - self.length] = 0
        elif self.step > 0:
            mix_mask[-self.step:] = 0
        
        for i, p in enumerate(self.mix_policies):
            if not torch.any(mix_mask == i):
                continue
            out_mask = torch.zeros((self.length - 1, 2), device=self.cent_player.device, dtype=torch.bool)
            out_mask[:, self.player_id] = (mix_mask == i)
            x = p.get_actions(
                share_obs[mix_mask == i],
                obs[mix_mask == i],
                self.cent_player.turn_mp_rnn_states[out_mask],
                self.cent_player.turn_mp_rnn_states_critic[out_mask],
                self.cent_player.turn_mp_masks[out_mask],
                available_actions[mix_mask == i]
            )
            # print(x[1].flatten())
            # print(self.turn_mp_actions[out_mask].flatten())
            (
                self.cent_player.turn_mp_values[out_mask],
                _,
                self.cent_player.turn_mp_action_log_probs[out_mask],
                self.cent_player.turn_mp_rnn_states[out_mask],
                self.cent_player.turn_mp_rnn_states_critic[out_mask]
            ) = x

            self.cent_player.turn_mp_actions[out_mask] = x[1].to(torch.float)
            # print(self.turn_mp_values[out_mask].flatten())

        # (action, action_log_prob, rnn_state) = self.actor(
        #     obs,
        #     self.cent_player.turn_mp_rnn_states[:, self.player_id],
        #     self.cent_player.turn_mp_masks[:, self.player_id],
        #     available_actions,
        # )
        # critic = self.cent_player.trainer.policy.critic
        # (value, rnn_state_critic) = critic(
        #     share_obs,
        #     self.cent_player.turn_mp_rnn_states_critic[:, self.player_id],
        #     self.cent_player.turn_mp_masks[:, self.player_id],
        # )

        self.cent_player.turn_mp_obs[:, self.player_id] = obs
        self.cent_player.turn_mp_share_obs[:, self.player_id] = share_obs
        self.cent_player.turn_mp_available_actions[:, self.player_id] = available_actions
        # self.cent_player.turn_mp_values[:, self.player_id] = value
        # self.cent_player.turn_mp_actions[:, self.player_id] = action
        # self.cent_player.turn_mp_action_log_probs[:, self.player_id] = action_log_prob
        # self.cent_player.turn_mp_rnn_states[:, self.player_id] = rnn_state
        # self.cent_player.turn_mp_rnn_states_critic[:, self.player_id] = rnn_state_critic
        self.cent_player.turn_mp_rewards[:, self.player_id] = 0
        self.cent_player.turn_mp_active_masks[:, self.player_id] = 1

        self.step += 1
        if self.step > 2 * self.length:
            self.step = 0

        return self.cent_player.turn_mp_actions[:, self.player_id]

    def update(self, rewards, dones):
        rewards = rewards
        dones = dones.to(torch.bool)
        # print(dones)
        # print(rewards, self.cent_player.turn_mp_rewards[:, 1])
        self.cent_player.turn_mp_rewards[:, self.player_id] += rewards[:, None]

        self.cent_player.turn_mp_masks[dones, self.player_id] = 0
        self.cent_player.turn_mp_rnn_states[dones, self.player_id] = 0
        self.cent_player.turn_mp_rnn_states_critic[dones, self.player_id] = 0

        self.cent_player.turn_mp_masks[~dones, self.player_id] = 1
        
class DecentralizedAgent(VectorAgent):
    def __init__(self, cent_player: MainPlayer, player_id: int, policy=None):
        self.cent_player = cent_player
        self.player_id = player_id
        if policy is None:
            self.actor = self.cent_player.trainer.policy.actor
        else:
            self.actor = policy

        self.rnn_states = torch.zeros(1)
        self.masks = torch.ones(1)

    def get_action(self, obs, record=True):
        available_actions = obs.action_mask
        share_obs = obs.state
        # choose = obs.active
        obs = obs.obs

        (action, action_log_prob, rnn_state) = self.actor(
            obs,
            self.rnn_states,
            self.masks,
            available_actions,
        )

        return action

    def update(self, rewards, dones):
        pass

class TFJSAgent(VectorAgent):
    def __init__(self, read_file: str, player_id: int):
        self.model = torch.load(read_file).cuda()

    def get_action(self, obs, record=True):
        return torch.distributions.Categorical(self.model(obs.obs.to(torch.float))).sample().unsqueeze(-1).cpu()

    def update(self, rewards, dones):
        pass

# class DecentralizedAgent(Agent):
#     def __init__(self, policy, critic=None):
#         self.actor = policy
#         self.critic = critic
#         self.rnn_states = torch.zeros(1)
#         self.masks = torch.ones(1)

#     def get_action(self, obs, record=True, deterministic=False):
#         obs, available_actions = obs

#         (action, _, rnn_state) = self.actor(
#             obs,
#             self.rnn_states,
#             self.masks,
#             available_actions,
#             deterministic=deterministic
#         )
#         self.rnn_states = rnn_state
#         return _t2n(action)

#     def get_value(self, obs):
#         obs, available_actions = obs

#         values, _ = self.critic(obs, self.rnn_states, self.masks)
#         return _t2n(values)

#     def predict(self, observation, record=True, deterministic=False):
#         return self.get_action((observation, None), record, deterministic)

#     def update(self, reward, done):
#         pass


# class BCAgent(Agent):
#     def __init__(self, fload):
#         self.base_policy = reconstruct_policy(fload)

#     def get_action(self, obs, record=True, deterministic=False):
#         obs, available_actions = obs

#         # pi_features, vf_features = self.base_policy.extract_features(torch.tensor(obs))

#         # print(pi_features, vf_features)

#         act, _, log_probs = action_from_policy(obs, self.base_policy)
#         # print(log_probs, act)
#         # print(act, log_probs)

#         # log_probs[~available_actions] = -float('inf')

#         # if deterministic:
#         #     return torch.argmax(log_probs)

#         # return torch.distributions.Categorical(log_probs).sample()
#         # print(act)
#         return act

#     def get_value(self, obs):
#         return 0

#     def predict(self, observation, record=True, deterministic=False):
#         return self.get_action((observation, None), record, deterministic)

#     def update(self, reward, done):
#         pass


# class MixedAgent(Agent):
#     def __init__(
#         self,
#         cent_player: MainPlayer,
#         player_id: int,
#         other_policy: R_Actor,
#         p_other=0.5,
#     ):
#         self.cent_player = cent_player
#         self.player_id = player_id
#         self.other_policy = other_policy
#         self.p_other = p_other

#         self.second_phase = False

#     def get_action(self, obs, record=True):
#         obs, share_obs, available_actions = obs
#         self.cent_player.trainer.prep_rollout()

#         if self.second_phase or np.random.rand() > self.p_other:
#             policy = self.cent_player.trainer.policy
#         else:
#             policy = self.other_policy

#         if self.second_phase:
#             (
#                 value,
#                 action,
#                 action_log_prob,
#                 rnn_state,
#                 rnn_state_critic,
#             ) = self.cent_player.trainer.policy.get_actions(
#                 share_obs,
#                 obs,
#                 self.cent_player.turn_rnn_states[0, self.player_id],
#                 self.cent_player.turn_rnn_states_critic[0, self.player_id],
#                 self.cent_player.turn_masks[0, self.player_id],
#                 available_actions,
#             )
#         else:
#             actor = (
#                 self.other_policy
#                 if np.random.rand() > self.p_other
#                 else self.cent_player.trainer.policy.actor
#             )
#             (action, action_log_prob, rnn_state) = actor(
#                 obs,
#                 self.cent_player.turn_rnn_states[0, self.player_id],
#                 self.cent_player.turn_masks[0, self.player_id],
#                 available_actions,
#             )
#             critic = self.cent_player.trainer.policy.critic
#             (value, rnn_state_critic) = critic(
#                 share_obs,
#                 self.cent_player.turn_rnn_states_critic[0, self.player_id],
#                 self.cent_player.turn_masks[0, self.player_id],
#             )

#         if record:
#             self.cent_player.turn_obs[0, self.player_id] = obs.copy()
#             self.cent_player.turn_share_obs[0, self.player_id] = share_obs.copy()
#             self.cent_player.turn_available_actions[
#                 0, self.player_id
#             ] = available_actions.copy()
#             self.cent_player.turn_values[0, self.player_id] = _t2n(value)
#             self.cent_player.turn_actions[0, self.player_id] = _t2n(action)
#             self.cent_player.turn_action_log_probs[0, self.player_id] = _t2n(
#                 action_log_prob
#             )
#             self.cent_player.turn_rnn_states[0, self.player_id] = _t2n(rnn_state)
#             self.cent_player.turn_rnn_states_critic[0, self.player_id] = _t2n(
#                 rnn_state_critic
#             )
#             self.cent_player.turn_rewards[0, self.player_id] = 0
#             self.cent_player.turn_active_masks[0, self.player_id] = 1
#         return _t2n(action)

#     def update(self, reward, done):
#         self.cent_player.turn_rewards[0, self.player_id] += reward

#         if done:
#             self.cent_player.turn_masks[0, self.player_id] = 0
#             self.cent_player.turn_rnn_states[0, self.player_id] = 0
#             self.cent_player.turn_rnn_states_critic[0, self.player_id] = 0
#         else:
#             self.cent_player.turn_masks[0, self.player_id] = 1
