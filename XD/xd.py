import numpy as np
import torch
import torch.nn as nn
from MAPPO.utils.util import get_gard_norm, huber_loss, mse_loss, check
from MAPPO.utils.valuenorm import ValueNorm

from collections import defaultdict

from .MCPolicy import MCPolicy


class XD:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model,
                 policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(
        self,
        args,
        policy: MCPolicy,
        agent_set,
        xp_weight,
        mp_weight,
        use_average,
        device=torch.device("cpu"),
    ):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.agent_set = agent_set
        self.xp_weight = xp_weight
        self.mp_weight = mp_weight
        self.use_average = use_average

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        assert not (
            self._use_popart and self._use_valuenorm
        ), "self._use_popart and self._use_valuenorm can not be set \
            True simultaneously"

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, active_masks_batch
    ):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value predictions
              from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is
              active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = (
                self.value_normalizer.normalize(return_batch) - value_pred_clipped
            )
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (
                value_loss * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, dir_weight=1.0, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv) * dir_weight

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch,
            obs_batch,
            rnn_states_batch,
            rnn_states_critic_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self._use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        #######################################################
        # Multiply policy_loss for sp/mp updates
        #######################################################
        policy_loss = policy_action_loss  # * dir_weight

        self.policy.actor_optimizer.zero_grad()
        actor_loss = policy_loss - dist_entropy * self.entropy_coef
        # if update_actor:
        #     (policy_loss - dist_entropy * self.entropy_coef).backward()

        # if self._use_max_grad_norm:
        #     actor_grad_norm = nn.utils.clip_grad_norm_(
        #         self.policy.actor.parameters(), self.max_grad_norm
        #     )
        # else:
        #     actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        # self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(
            values, value_preds_batch, return_batch, active_masks_batch
        )

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return (
            value_loss,
            critic_grad_norm,
            policy_loss,
            dist_entropy,
            # actor_grad_norm,
            imp_weights,
        ), actor_loss

    def calc_advantanges(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(
                buffer.value_preds[:-1]
            )
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        return advantages

    def get_gen(self, buffer, advantages):
        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(
                advantages, self.num_mini_batch, self.data_chunk_length
            )
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(
                advantages, self.num_mini_batch
            )
        else:
            data_generator = buffer.feed_forward_generator(
                advantages, self.num_mini_batch
            )
        return data_generator

    def get_partial_gen(self, buffer, advantages, idx):
        if self._use_recurrent_policy:
            data_generator = buffer.partial_recurrent_generator(
                advantages, idx, self.num_mini_batch, self.data_chunk_length
            )
        elif self._use_naive_recurrent:
            data_generator = buffer.partial_naive_recurrent_generator(
                advantages, idx, self.num_mini_batch
            )
        else:
            data_generator = buffer.partial_ff_generator(
                advantages, idx, self.num_mini_batch
            )
        return data_generator

    def train_step(self, data_generator, train_info, weight, update_actor=True):
        for sample in data_generator:
            (
                value_loss,
                critic_grad_norm,
                policy_loss,
                dist_entropy,
                # actor_grad_norm,
                imp_weights,
            ), actor_loss = self.ppo_update(sample, weight, update_actor)

            train_info["value_loss"] += value_loss.item()
            train_info["policy_loss"] += policy_loss.item()
            train_info["dist_entropy"] += dist_entropy.item()
            train_info["actor_grad_norm"] += 0  # actor_grad_norm.item()
            train_info["critic_grad_norm"] += critic_grad_norm.item()
            train_info["ratio"] += imp_weights.mean().item()
            return actor_loss

    def get_best(self, bufs0, bufs1):
        means = np.zeros(len(bufs0))
        for i, (bi0, bi1) in enumerate(zip(bufs0, bufs1)):
            means[i] = np.mean(bi0.rewards) + np.mean(bi1.rewards)

        return np.argmax(means)

    def train(self, sp_buf, xp_buf0, xp_buf1, mp_buf, update_actor=True, best_i=None):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training
                update (e.g. loss, grad norms, etc).
        """
        sp_adv = self.calc_advantanges(sp_buf)
        xp0_adv = []
        xp1_adv = []
        mp_adv = []
        for i in range(len(self.agent_set)):
            xp0_adv.append(self.calc_advantanges(xp_buf0[i]))
            xp1_adv.append(self.calc_advantanges(xp_buf1[i]))
            mp_adv.append(self.calc_advantanges(mp_buf[i]))

        train_info = defaultdict(lambda: 0)
        if len(self.agent_set) > 0 and best_i is None:
            best_i = self.get_best(xp_buf0, xp_buf1)
        for _ in range(self.ppo_epoch):
            self.policy.set_sp()
            loss = self.train_step(self.get_gen(sp_buf, sp_adv), train_info, 1, update_actor)
            if self.xp_weight != 0:
                if self.use_average:
                    for i in range(len(self.agent_set)):
                        self.policy.set_xp(0, i)
                        loss += self.train_step(
                            self.get_partial_gen(xp_buf0[i], xp0_adv[i], 0),
                            train_info,
                            -self.xp_weight,
                            update_actor,
                        )

                        self.policy.set_xp(1, i)
                        loss += self.train_step(
                            self.get_partial_gen(xp_buf1[i], xp1_adv[i], 1),
                            train_info,
                            -self.xp_weight,
                            update_actor,
                        )
                elif len(self.agent_set) > 0:
                    i = best_i
                    self.policy.set_xp(0, i)
                    loss += self.train_step(
                        self.get_partial_gen(xp_buf0[i], xp0_adv[i], 0),
                        train_info,
                        -self.xp_weight,
                        update_actor,
                    )

                    i = best_i
                    self.policy.set_xp(1, i)
                    loss += self.train_step(
                        self.get_partial_gen(xp_buf1[i], xp1_adv[i], 1),
                        train_info,
                        -self.xp_weight,
                        update_actor,
                    )
            if self.mp_weight != 0 and len(self.agent_set) > 0:
                self.policy.set_mp()
                i = best_i
                loss += self.train_step(
                    self.get_gen(mp_buf[i], mp_adv[i]),
                    train_info,
                    self.mp_weight / len(self.agent_set),
                    update_actor,
                )
            loss.backward()

            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.max_grad_norm
                )
            else:
                actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            train_info["actor_grad_norm"] += actor_grad_norm.item()
            self.policy.actor_optimizer.step()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info["best_i"] = best_i
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        for crit in (
            [self.policy.critic] + self.policy.xp_critic0 + self.policy.xp_critic1
        ):
            crit.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        for crit in (
            [self.policy.critic] + self.policy.xp_critic0 + self.policy.xp_critic1
        ):
            crit.eval()
