from abc import ABC, abstractmethod
from .vectorobservation import VectorObservation

import torch
import torch.nn as nn

from typing import Optional

class VectorAgent(ABC):
    @abstractmethod
    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        """
        Return an action given an observation.
        :param obs: The observation to use
        :param record: Whether to record the obs, action pair (for training)
        :returns: The action to take
        """

    @abstractmethod
    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """
        Add new rewards and done information if the agent can learn.
        Each update corresponds to the most recent `get_action` (where
        `record` is True). If there are multiple calls to `update` that
        correspond to the same `get_action`, their rewards are summed up and
        the last done flag will be used.
        :param reward: The rewards receieved from the previous action step
        :param done: Whether the game is done
        """


class RandomVectorAgent(VectorAgent):
    def __init__(self, sampler):
        self.sampler = sampler
        
    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        return self.sampler()

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        return












import time
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CleanRLNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.share_observation_space.shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=0.01),
        )
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, envs.action_space.n), std=0.01),
        )
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.share_observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        
        # self.actor = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, envs.action_space.n), std=0.01),
        # )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, state, action_mask, action=None):
        logits = self.actor(x)
        logits[torch.logical_not(action_mask)] = -float('inf')
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(state)


class CleanPPOAgent(VectorAgent):
    def __init__(self,
                 envs: "VectorMultiAgentEnv",
                 name: str,
                 device: torch.device,
                 num_updates: int,
                 verbose: bool = True,
                 lr: float = 2.5e-4,
                 num_steps: int = 128,
                 anneal_lr: bool = True,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 num_minibatches: int = 4,
                 update_epochs: int = 4,
                 norm_adv: bool = True,
                 clip_coef: float = 0.2,
                 clip_vloss: bool = True,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 target_kl: Optional[float] = None):
        self.envs = envs
        self.num_envs = envs.num_envs
        self.name = name
        self.device = device
        self.verbose = verbose

        self.lr = lr
        self.num_steps = num_steps
        self.anneal_lr = anneal_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_minibatches = num_minibatches
        self.update_epochs = update_epochs
        self.norm_adv = norm_adv
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)

        if self.verbose:
            self.writer = SummaryWriter(f"runs/{name}")
            self.writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self).items()])),
            )

        self.agent = CleanRLNetwork(self.envs).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr, eps=1e-5)

        self.obs = torch.zeros((self.num_steps, self.num_envs) + envs.observation_space.shape).to(device)
        self.states = torch.zeros((self.num_steps, self.num_envs) + envs.share_observation_space.shape).to(device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + envs.action_space.shape).to(device)
        self.action_masks = torch.zeros((self.num_steps, self.num_envs, envs.action_space.n)).to(device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(device)
        self.active = torch.zeros((self.num_steps, self.num_envs), dtype=torch.bool).to(device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(device)

        self.global_step = 0
        self.step = 0
        self.start_time = time.time()
        self.num_updates = num_updates
        self.updates = 1

        self.next_done = torch.zeros(self.num_envs, dtype=torch.bool).to(device)
        self.new_game = torch.zeros(self.num_envs, dtype=torch.bool).to(device)

        self.running_rewards = torch.zeros(self.num_envs).to(device)
        self.last_active = torch.zeros(self.num_envs, dtype=torch.long).to(device)

        self.mean_return_sum = 0
        self.num_returns = 0

    def update(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        dones = dones.to(dtype=torch.bool)
        # multi-agent decisions:
        # - rewards from before an agent's first action are discarded for returns calculation but kept for running rewards
        # - continue rewards from previous timestep (and accumulate) if not active
        self.running_rewards += rewards
        self.rewards[self.last_active] += torch.where(self.new_game, 0, rewards.view(-1))
        # print(self.rewards)
        self.next_done |= dones

        if torch.any(dones):
            if self.verbose:
                # self.writer.add_scalar("charts/episodic_return", torch.mean(self.running_rewards[dones]), self.global_step)
                # print(torch.mean(self.running_rewards[dones]))
                self.writer.add_scalar("charts/min_episodic_return", torch.min(self.running_rewards[dones]), self.global_step)
                self.writer.add_scalar("charts/max_episodic_return", torch.max(self.running_rewards[dones]), self.global_step)
            self.mean_return_sum += torch.mean(self.running_rewards[dones])
            self.num_returns += 1
            self.running_rewards[dones] = 0.0
            self.new_game[dones] = True
            
        self.step += 1
        self.global_step += 1

    def get_action(self, obs: VectorObservation, record: bool = True) -> torch.tensor:
        if self.global_step > 0 and self.global_step % self.num_steps == 0 and record:
            self.step = 0

            if self.anneal_lr:
                frac = 1.0 - (self.updates - 1.0) / self.num_updates
                lrnow = frac * self.lr
                self.optimizer.param_groups[0]["lr"] = lrnow
            
            with torch.no_grad():
                next_value = self.agent.get_value(obs.state.float()).reshape(-1)
                advantages = torch.zeros_like(self.rewards).to(self.device)

                delta = torch.zeros(self.num_envs).to(self.device)

                bootstrapped = obs.active.detach().clone().to(dtype=torch.bool)
                nextnonterminal = torch.zeros(self.num_envs).to(self.device)
                nextvalues = torch.zeros(self.num_envs).to(self.device)
                
                lastgaelam = torch.zeros(self.num_envs).to(self.device)
                nextnonterminal[bootstrapped] = 1.0 - self.next_done[bootstrapped].to(torch.float)
                nextvalues[bootstrapped] = next_value[bootstrapped]
                
                for t in reversed(range(self.num_steps)):
                    mask = self.active[t]
                    computemask = mask
                    
                    if not torch.all(bootstrapped):
                        bootmask = torch.logical_and(mask, bootstrapped.logical_not())
                        computemask = mask & bootstrapped.logical_not()

                        bootstrapped |= mask

                        #disable advantages for these final bootstrapped values
                        self.active[t, bootmask] = False
                    
                    delta[computemask] = self.rewards[t, computemask] + self.gamma * nextvalues[computemask] * nextnonterminal[computemask] - self.values[t, computemask]
                    advantages[t, computemask] = lastgaelam[computemask] = delta[computemask] + self.gamma * self.gae_lambda * nextnonterminal[computemask] * lastgaelam[computemask]

                    nextnonterminal[mask] = 1.0 - self.dones[t, mask]
                    nextvalues[mask] = self.values[t, mask]
                returns = advantages + self.values

            # print("dones:", self.dones)
            # print("returns:", returns)

            # flatten the batch
            b_obs = self.obs[self.active].reshape((-1,) + self.envs.observation_space.shape)
            b_logprobs = self.logprobs[self.active].reshape(-1)
            b_actions = self.actions[self.active].reshape((-1,) + self.envs.action_space.shape)
            b_advantages = advantages[self.active].reshape(-1)
            b_returns = returns[self.active].reshape(-1)
            b_values = self.values[self.active].reshape(-1)

            b_states = self.states[self.active].reshape((-1,) + self.envs.share_observation_space.shape)
            b_action_masks = self.action_masks[self.active].reshape((-1, self.envs.action_space.n))

            clipfracs = []
            for epoch in range(self.update_epochs):
                # todo: minibatches?
                mb_inds = torch.randperm(b_values.size(0))

                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_states[mb_inds], b_action_masks[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if self.verbose:
                if self.num_returns != 0:
                    self.writer.add_scalar("charts/episodic_return", self.mean_return_sum/self.num_returns, self.global_step)
                    self.mean_return_sum = 0
                    self.num_returns = 0
                
                self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
                self.writer.add_scalar("losses/value_loss", v_loss.item(), self.global_step)
                self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.global_step)
                self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.global_step)
                self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.global_step)
                self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.global_step)
                self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.global_step)
                self.writer.add_scalar("losses/explained_variance", explained_var, self.global_step)
            
                # print("SPS:", int(self.global_step / (time.time() - self.start_time)))
                self.writer.add_scalar("charts/SPS", int(self.global_step / (time.time() - self.start_time)), self.global_step)
            self.updates += 1

        with torch.no_grad():
            action, logprob, _, value = self.agent.get_action_and_value(obs.obs.float(), obs.state.float(), obs.action_mask)
            if record:
                self.values[self.step] = value.flatten()

        if record:
            self.obs[self.step] = obs.obs
            self.dones[self.step] = self.next_done
            self.states[self.step] = obs.state
            self.active[self.step] = obs.active
            # self.values[self.step] = value.flatten()
            self.actions[self.step] = action
            self.action_masks[self.step] = obs.action_mask
            self.logprobs[self.step] = logprob

            self.next_done[:] = False
            self.rewards[self.step] = 0

            self.last_active[obs.active] = self.step
            self.new_game[obs.active] = False
        return action[:,None]


    
