import torch
from MAPPO.r_actor_critic import R_Critic
from MAPPO.utils.util import update_linear_schedule
from MAPPO.rMAPPOPolicy import R_MAPPOPolicy


class MCPolicy(R_MAPPOPolicy):
    def __init__(
        self,
        args,
        obs_space,
        cent_obs_space,
        act_space,
        num_priors,
        device=torch.device("cpu"),
    ):
        super().__init__(args, obs_space, cent_obs_space, act_space, device)
        self.sp_critic = self.critic
        self.sp_critic_optimizer = self.critic_optimizer

        self.mp_critic = R_Critic(args, self.share_obs_space, self.device)
        self.mp_critic_optimizer = torch.optim.Adam(
            self.mp_critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

        self.xp_critic0 = [
            R_Critic(args, self.share_obs_space, self.device) for _ in range(num_priors)
        ]
        self.xp_critic1 = [
            R_Critic(args, self.share_obs_space, self.device) for _ in range(num_priors)
        ]

        self.xp_critic0_optimizer = [
            torch.optim.Adam(
                self.xp_critic0[i].parameters(),
                lr=self.critic_lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
            for i in range(num_priors)
        ]
        self.xp_critic1_optimizer = [
            torch.optim.Adam(
                self.xp_critic1[i].parameters(),
                lr=self.critic_lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
            for i in range(num_priors)
        ]

    def set_sp(self):
        self.critic = self.sp_critic
        self.critic_optimizer = self.sp_critic_optimizer

    def set_mp(self):
        self.critic = self.mp_critic
        self.critic_optimizer = self.mp_critic_optimizer

    def set_xp(self, ego_id, other_convention):
        if ego_id == 0:
            self.critic = self.xp_critic0[other_convention]
            self.critic_optimizer = self.xp_critic0_optimizer[other_convention]
        else:
            self.critic = self.xp_critic1[other_convention]
            self.critic_optimizer = self.xp_critic1_optimizer[other_convention]

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(
            self.sp_critic_optimizer, episode, episodes, self.critic_lr
        )

        for crit in self.xp_critic0_optimizer + self.xp_critic1_optimizer:
            update_linear_schedule(crit, episode, episodes, self.critic_lr)
