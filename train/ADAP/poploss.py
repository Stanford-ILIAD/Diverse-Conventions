from itertools import permutations

import torch as th
from torch.distributions import kl


class PopulationLoss:
    def get_population_loss(self, policies, train_batch):
        return 0


class ADAPLoss(PopulationLoss):
    def __init__(self, losscoef):
        self.losscoef = losscoef

    def get_population_loss(self, policies, train_batch):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,\
            actions_batch, value_preds_batch, return_batch, masks_batch, \
            active_masks_batch, old_action_log_probs_batch, adv_targ, \
            available_actions_batch = train_batch

        all_action_dists = []
        # print("Start collecting per policy")
        for policy in policies:
            action_logits = policy.actor.get_logits(obs_batch,
                                                    rnn_states_batch,
                                                    masks_batch,
                                                    available_actions_batch,
                                                    active_masks_batch)
            all_action_dists.append(action_logits)
        # print("End collecting per policy")
        all_CLs = [th.mean(th.exp(-kl.kl_divergence(a, b)))
                   for a, b in permutations(all_action_dists, 2)]
        # print("End computations")
        rawans = sum(all_CLs)/len(all_CLs)
        return rawans * self.losscoef
