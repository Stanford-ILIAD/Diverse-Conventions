"""
This file contains agents that are specific to the Tree environment.
"""
import sys
import numpy as np

from pantheonrl.common.agents import Agent

from .tree_env import PantheonTree, to_mult_bin, DEPTH


def get_move_string(obs: np.ndarray) -> str:
    """Converts obs into human-readable history"""
    ans = ''
    for i in reversed(range(np.argwhere(obs).max())):
        ans += 'R' if obs[i] else 'L'
    return ans


class TreeUser(Agent):
    """
    Agent representing human user input to the tree environment
    """

    def __init__(self, env: PantheonTree):
        self.saw_tutorial = False
        self.env = env

    def print_tutorial(self):
        """Prints explanation of environment"""
        print("""
        In the Tree environment, you have two choices at each
        timestep: go left (l) or go right (r). You have to make 3
        moves in total for a single game.

        To get a reward of 1: Player 0 must move l on the first
        action, and player 1 must move l on the last action. The
        agents can move l or r for any of the other moves.

        To get a reward of 3:
        If you are player 0, you must move RLL
        If you are player 1, you must move LLR
        """)

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        if not self.saw_tutorial:
            self.print_tutorial()
            self.saw_tutorial = True
        obs = obs[0]  # remove mask

        print("You are player", obs[-1])
        print("Your past actions are:", get_move_string(obs[:-1]))
        move = -1
        while move == -1:
            action = input('Next move [l/r/(q)uit]: ').lower()
            if action == 'l':
                move = 0
            elif action == 'r':
                move = 1
            elif action in ('q', 'quit'):
                print("Bye!")
                sys.exit(0)
            else:
                print("Invalid move!")
        return np.array([move])

    def update(self, reward: float, done: bool) -> None:
        if done:
            print("Final state is",
                  get_move_string(to_mult_bin(self.env.state)))


class RandomTreeAgent(Agent):
    """Agent representing random actor"""

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        return np.random.randint(0, 2, (1))

    def update(self, reward: float, done: bool) -> None:
        pass


class SafeAgent(Agent):
    """Agent that gets a reward of 1"""

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        obs = obs[0]
        player_num = obs[-1]
        obs = obs[:-1]
        time = np.argwhere(obs).max()
        if (not player_num and time == 0) or (player_num and time == DEPTH // 2 - 1):
            return np.array([0])
        return np.random.randint(0, 2, (1))

    def update(self, reward: float, done: bool) -> None:
        pass


class RiskyAgent(Agent):
    """Agent that gets a reward of 3"""

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        obs = obs[0]
        player_num = obs[-1]
        obs = obs[:-1]
        time = np.argwhere(obs).max()
        if (not player_num and time == 0) or (player_num and time == DEPTH // 2 - 1):
            return np.array([1])
        return np.array([0])

    def update(self, reward: float, done: bool) -> None:
        pass
