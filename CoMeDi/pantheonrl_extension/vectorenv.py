from abc import ABC, abstractmethod
from .vectorobservation import VectorObservation
from .vectoragent import VectorAgent
from dataclasses import dataclass

from typing import Optional, List, Tuple

import numpy as np

import gym
import torch

class PlayerException(Exception):
    """ Raise when players in the environment are incorrectly set """


@dataclass
class DummyEnv():
    """
    Environment representing a partner agent's observation and action space.
    """
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space


class VectorMultiAgentEnv(ABC):
    """
    Base class for all Multi-agent environments.

    :param ego_ind: The player that the ego represents
    :param n_players: The number of players in the game
    :param resample_policy: The resampling policy to use
    - (see set_resample_policy)
    :param partners: Lists of agents to choose from for the partner players
    :param ego_extractor: Function to extract Observation into the type the
        ego expects
    """

    def __init__(self,
                 num_envs: int,
                 device: torch.device,
                 ego_ind: int = 0,
                 n_players: int = 2,
                 resample_policy: str = "default",
                 partners: Optional[List[List[VectorAgent]]] = None):
        self.num_envs = num_envs
        self.device = device
        
        self.ego_ind = ego_ind
        self.n_players = n_players
        if partners is not None:
            if len(partners) != n_players - 1:
                raise PlayerException(
                    "The number of partners needs to equal the number \
                    of non-ego players")

            for plist in partners:
                if not isinstance(plist, list) or not plist:
                    raise PlayerException(
                        "Sublist for each partner must be nonempty list")

        self.partners = partners or [[]] * (n_players - 1)
        self.partnerids = [0] * (n_players - 1)

        self._obs: Tuple[Optional[np.ndarray], ...] = tuple()

        # self._dones = torch.zeros((n_players, num_envs), dtype=torch.bool, device=device)
        # self._alives = torch.ones((n_players, num_envs), dtype=torch.bool, device=device)
        self._actions = None
        self.set_resample_policy(resample_policy)

    def getDummyEnv(self, player_num: int):
        """
        Returns a dummy environment with just an observation and action
        space that a partner agent can use to construct their policy network.

        :param player_num: the partner number to query
        """
        return self

    def _get_partner_num(self, player_num: int) -> int:
        if player_num == self.ego_ind:
            raise PlayerException(
                "Ego agent is not set by the environment")
        elif player_num > self.ego_ind:
            return player_num - 1
        return player_num

    def add_partner_agent(self, agent: VectorAgent, player_num: int = 1) -> None:
        """
        Add agent to the list of potential partner agents. If there are
        multiple agents that can be a specific player number, the environment
        randomly samples from them at the start of every episode.

        :param agent: VectorAgent to add
        :param player_num: the player number that this new agent can be
        """
        self.partners[self._get_partner_num(player_num)].append(agent)

    def set_partnerid(self, agent_id: int, player_num: int = 1) -> None:
        """
        Set the current partner agent to use

        :param agent_id: agent_id to use as current partner
        """
        partner_num = self._get_partner_num(player_num)
        assert(agent_id >= 0 and agent_id < len(self.partners[partner_num]))
        self.partnerids[partner_num] = agent_id

    def resample_random(self) -> None:
        """ Randomly resamples each partner policy """
        self.partnerids = [np.random.randint(len(plist))
                           for plist in self.partners]

    def resample_round_robin(self) -> None:
        """
        Sets the partner policy to the next option on the list for round-robin
        sampling.

        Note: This function is only valid for 2-player environments
        """
        self.partnerids = [(self.partnerids[0] + 1) % len(self.partners[0])]

    def set_resample_policy(self, resample_policy: str) -> None:
        """
        Set the resample_partner method to round "robin" or "random"

        :param resample_policy: The new resampling policy to use.
        - Valid values are: "default", "robin", "random"
        """
        if resample_policy == "default":
            resample_policy = "robin" if self.n_players == 2 else "random"

        if resample_policy == "robin" and self.n_players != 2:
            raise PlayerException(
                "Cannot do round robin resampling for >2 players")

        if resample_policy == "robin":
            self.resample_partner = self.resample_round_robin
        elif resample_policy == "random":
            self.resample_partner = self.resample_random
        else:
            raise PlayerException(
                f"Invalid resampling policy: {resample_policy}")

    def _get_actions(self, obs, ego_act=None):
        actions = []
        for player, ob in zip(range(self.n_players), obs):
            if player == self.ego_ind:
                actions.append(ego_act)
            else:
                p = self._get_partner_num(player)
                agent = self.partners[p][self.partnerids[p]]
                actions.append(agent.get_action(ob))
        if self._actions is None:
            self._actions = torch.stack(actions)
        else:
            torch.stack(actions, out=self._actions)
        return self._actions

    def _update_players(self, rews, done):
        # self._dones = (self._dones & torch.logical_not(self._alives)) | done
        # self._alives = torch.stack([o.active for o in self._obs], out=self._alives)
        
        for i in range(self.n_players - 1):
            playernum = i + (0 if i < self.ego_ind else 1)
            nextrew = rews[playernum]
            nextdone = done  # self._dones[playernum]
            self.partners[i][self.partnerids[i]].update(nextrew, nextdone)
        
    def step(
                self,
                action: torch.Tensor
            ):
        """
        Run one timestep from the perspective of the ego-agent. This involves
        calling the ego_step function and the alt_step function to get to the
        next observation of the ego agent.

        Accepts the ego-agent's action and returns a tuple of (observation,
        reward, done, info) from the perspective of the ego agent.

        :param action: An action provided by the ego-agent.

        :returns:
            observation: Ego-agent's next observation
            reward: Amount of reward returned after previous action
            done: Whether the episode has ended
            info: Extra information about the environment
        """

        acts = self._get_actions(self._obs, action)
        self._obs, rews, done, info = self.n_step(acts)
        
        self._update_players(rews, done)

        ego_obs = self._obs[self.ego_ind]
        ego_rew = rews[self.ego_ind]
        ego_done = done  # self._dones[self.ego_ind]
        return ego_obs, ego_rew, ego_done, info

    def reset(self):
        """
        Reset environment to an initial state and return the first observation
        for the ego agent.

        :returns: Ego-agent's first observation
        """
        self.resample_partner()
        self._obs = self.n_reset()

        # self._alives.fill_(1)
        # self._dones.fill_(0)

        ego_obs = self._obs[self.ego_ind]

        return ego_obs

    @abstractmethod
    def n_step(
                    self,
                    actions: torch.Tensor,
                ):
        """
        Perform the actions specified by the agents that will move. This
        function returns a tuple of (next agents, observations, both rewards,
        done, info).

        This function is called by the `step` function.

        :param actions: List of action provided agents that are acting on this
        step.

        :returns:
            observations: List representing the next VectorObservations
            rewards: Tensor representing the rewards of all agents: num_agents x num_envs
            done: Whether the episodes have ended
            info: Extra information about the environment
        """

    @abstractmethod
    def n_reset(self):
        """
        Reset the environment and return which agents will move first along
        with their initial observations.

        This function is called by the `reset` function.

        :returns:
            observations: List of VectorObservations representing the observations of
                each agent
        """

    def close(self, **kwargs):
        pass


def to_torch(a):
    return a.detach().clone()


class MadronaEnv(VectorMultiAgentEnv):

    def __init__(self, num_envs, gpu_id, sim, debug_compile=True, obs_size=None, state_size=None, discrete_action_size=None, env_device=None):

        self.sim = sim

        self.static_dones = self.sim.done_tensor().to_torch()
        self.static_active_agents = self.sim.active_agent_tensor().to_torch()
        
        self.static_actions = self.sim.action_tensor().to_torch()
        # print(self.static_actions)
        self.static_observations = self.sim.observation_tensor().to_torch()
        self.static_agent_states = self.sim.agent_state_tensor().to_torch()
        self.static_action_masks = self.sim.action_mask_tensor().to_torch()
        self.static_rewards = self.sim.reward_tensor().to_torch()
        self.static_worldID = self.sim.world_id_tensor().to_torch().to(torch.long)
        self.static_agentID = self.sim.agent_id_tensor().to_torch().to(torch.long)
        
        self.obs_size = self.static_observations.shape[2] if obs_size is None else obs_size
        self.state_size = self.static_agent_states.shape[2] if state_size is None else state_size
        self.discrete_action_size = self.static_action_masks.shape[2] if discrete_action_size is None else discrete_action_size
        
        self.static_scattered_active_agents = self.static_active_agents.detach().clone()
        self.static_scattered_observations = self.static_observations.detach().clone()
        self.static_scattered_agent_states = self.static_agent_states.detach().clone()
        self.static_scattered_action_masks = self.static_action_masks.detach().clone()
        self.static_scattered_rewards = self.static_rewards.detach().clone()

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        if env_device is None:
            env_device = torch.device('cuda', gpu_id) if torch.cuda.is_available() else torch.device('cpu')
        
        super().__init__(num_envs, device=env_device, n_players=self.static_observations.shape[0])

        self.infos = [{}] * self.num_envs

    def to_torch(self, a):
        return a.to(self.device) #.detach().clone().to(self.device)

    def n_step(self, actions):
        actions_device = self.static_agentID.get_device()
        actions = actions.to(actions_device if actions_device != -1 else torch.device('cpu'))
        self.static_actions.copy_(actions[self.static_agentID, self.static_worldID, :])

        self.sim.step()

        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        obs = [VectorObservation(self.to_torch(self.static_scattered_active_agents[i].to(torch.bool)),
                                 self.to_torch(self.static_scattered_observations[i, :, :self.obs_size]),
                                 self.to_torch(self.static_scattered_agent_states[i, :, :self.state_size]),
                                 self.to_torch(self.static_scattered_action_masks[i, :, :self.discrete_action_size].to(torch.bool)))
               for i in range(self.n_players)]

        # print(obs[0].active, obs[1].active)
        # print(self.static_active_agents)
        # print(self.static_action_masks)

        return obs, self.to_torch(self.static_scattered_rewards), self.to_torch(self.static_dones), self.infos

    def n_reset(self):
        self.static_scattered_active_agents[self.static_agentID, self.static_worldID] = self.static_active_agents
        self.static_scattered_observations[self.static_agentID, self.static_worldID, :] = self.static_observations
        self.static_scattered_agent_states[self.static_agentID, self.static_worldID, :] = self.static_agent_states
        self.static_scattered_action_masks[self.static_agentID, self.static_worldID, :] = self.static_action_masks
        self.static_scattered_rewards[self.static_agentID, self.static_worldID] = self.static_rewards

        obs = [VectorObservation(self.to_torch(self.static_scattered_active_agents[i].to(torch.bool)),
                                 self.to_torch(self.static_scattered_observations[i, :, :self.obs_size]),
                                 self.to_torch(self.static_scattered_agent_states[i, :, :self.state_size]),
                                 self.to_torch(self.static_scattered_action_masks[i, :, :self.discrete_action_size].to(torch.bool)))
               for i in range(self.n_players)]
        return obs

    def close(self, **kwargs):
        pass

class SyncVectorEnv(VectorMultiAgentEnv):

    def __init__(self, env_fns, device=None):
        if device is None:
            device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
        self.envs = [fn() for fn in env_fns]
        
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.share_observation_space = self.envs[0].share_observation_space
        
        super().__init__(len(env_fns), device=device, n_players=self.envs[0].n_players)
        

    def n_step(self, actions):
        infos = []
        for i in range(len(self.envs)):
            envactions = []
            for agent in self.agents_tuples[i]:
                envactions.append(actions[agent, i])
            agentsi, obsi, rewsi, donesi, infosi = self.envs[i].n_step(tuple(envactions))
            if donesi:
                agentsi, obsi = self.envs[i].n_reset()
            self.agents_tuples[i] = agentsi
            if i == 0:
                self.static_rewards = torch.zeros((self.n_players, self.num_envs), device=self.device)
                self.static_dones = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
            infos.append(infosi)
            self.static_dones[i] = donesi
            for agent in range(self.n_players):
                self.static_rewards[agent, i] = rewsi[agent]

            for j in range(self.n_players):
                self.static_active_agents[j, i] = False
                
            for j in range(len(agentsi)):
                agent = agentsi[j]
                self.static_active_agents[agent, i] = True
                self.static_observations[agent, i] = torch.from_numpy(obsi[j][0])
                self.static_agent_states[agent, i] = torch.from_numpy(obsi[j][1])
                self.static_action_masks[agent, i] = torch.from_numpy(obsi[j][2])

        obs = [VectorObservation(to_torch(self.static_active_agents[i]),
                                 to_torch(self.static_observations[i]),
                                 to_torch(self.static_agent_states[i]),
                                 to_torch(self.static_action_masks[i]))
               for i in range(self.n_players)]
            
        return obs, self.static_rewards, self.static_dones, infos

    def n_reset(self):
        self.agents_tuples = []
        for i in range(len(self.envs)):
            agentsi, obsi = self.envs[i].n_reset()
            self.agents_tuples.append(agentsi)
            # obsi = (obs, state, action_mask)
            if i == 0:
                self.static_active_agents = torch.zeros((self.n_players, self.num_envs), device=self.device, dtype=torch.bool)
                self.static_observations = torch.zeros((self.n_players, self.num_envs) + obsi[0][0].shape, device=self.device)
                self.static_agent_states = torch.zeros((self.n_players, self.num_envs) + obsi[0][1].shape, device=self.device)
                self.static_action_masks = torch.ones((self.n_players, self.num_envs) + obsi[0][2].shape, device=self.device, dtype=torch.bool)

            for j in range(self.n_players):
                self.static_active_agents[j, i] = False

            for j in range(len(agentsi)):
                agent = agentsi[j]
                self.static_active_agents[agent, i] = True
                self.static_observations[agent, i] = torch.from_numpy(obsi[j][0])
                self.static_agent_states[agent, i] = torch.from_numpy(obsi[j][1])
                self.static_action_masks[agent, i] = torch.from_numpy(obsi[j][2])

        obs = [VectorObservation(to_torch(self.static_active_agents[i]),
                                 to_torch(self.static_observations[i]),
                                 to_torch(self.static_agent_states[i]),
                                 to_torch(self.static_action_masks[i]))
               for i in range(self.n_players)]
        return obs
