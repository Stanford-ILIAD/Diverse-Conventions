import multiprocessing as mp
from gym.vector.utils import CloudpickleWrapper

import torch
import sys

from .vectorenv import VectorMultiAgentEnv
from.vectorobservation import VectorObservation


def to_torch(a):
    return a  # .detach().clone()


class AsyncVectorEnv(VectorMultiAgentEnv):

    def __init__(self, env_fns, device=None, context=None, daemon=True):
        if device is None:
            device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')

        mp.set_start_method('spawn')
        ctx = mp.get_context(context)

        dummy_env = env_fns[0]()

        n_players = dummy_env.n_players

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()

        for idx, env_fn in enumerate(env_fns):
            parent_pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_worker,
                name=f"Worker<{type(self).__name__}>-{idx}",
                args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        self.error_queue,
                    )
            )
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

            process.daemon = daemon
            process.start()
            child_pipe.close()

        # self.envs = [fn() for fn in env_fns]

        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        self.share_observation_space = dummy_env.share_observation_space

        super().__init__(len(env_fns), device=device, n_players=n_players)


    def n_step(self, actions):
        for idx, pipe in enumerate(self.parent_pipes):
            pipe.send(("step", (self.agents_tuples[idx], actions[:, idx].cpu().numpy())))

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        agents, obs, rews, dones, infos = zip(*results)

        self.agents_tuples = []
        for i in range(self.num_envs):
            agentsi, obsi, rewsi, donesi = agents[i], obs[i], rews[i], dones[i]

            self.agents_tuples.append(agentsi)
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
        for pipe in self.parent_pipes:
            pipe.send(("reset", 0))

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        agents, obs = zip(*results)

        self.static_rewards = torch.zeros((self.n_players, self.num_envs), device=self.device)
        self.static_dones = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)

        self.static_active_agents = torch.zeros((self.n_players, self.num_envs), device=self.device, dtype=torch.bool)
        self.static_observations = torch.zeros((self.n_players, self.num_envs) + obs[0][0][0].shape, device=self.device)
        self.static_agent_states = torch.zeros((self.n_players, self.num_envs) + obs[0][0][1].shape, device=self.device)
        self.static_action_masks = torch.ones((self.n_players, self.num_envs) + obs[0][0][2].shape, device=self.device, dtype=torch.bool)

        self.agents_tuples = []
        for i in range(self.num_envs):
            agentsi, obsi = agents[i], obs[i]

            self.agents_tuples.append(agentsi)
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

    def close(self):
        for pipe in self.parent_pipes:
            if (pipe is not None) and (not pipe.closed):
                pipe.send(("close", None))
        for pipe in self.parent_pipes:
            if (pipe is not None) and (not pipe.closed):
                pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()


def _worker(index, env_fn, pipe, parent_pipe, error_queue):
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                agentsi, obsi = env.n_reset()
                pipe.send(((agentsi, obsi), True))
            elif command == "step":
                agents, actions = data
                envactions = []
                for agent in agents:
                    envactions.append(actions[agent])
                agentsi, obsi, rewsi, donesi, infosi = env.n_step(envactions)
                if donesi:
                    agentsi, obsi = env.n_reset()
                pipe.send(((agentsi, obsi, rewsi, donesi, infosi), True))
            elif command == "close":
                pipe.send((None, True))
                break
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception) as e:
        print("EXCEPTION", e)
        print(index, "BROKEN")
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
