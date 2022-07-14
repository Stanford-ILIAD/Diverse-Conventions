"""
This file contains agents that are specific to the Line environment.
"""
import sys
from typing import Tuple
import numpy as np

from pantheonrl.common.agents import Agent

from .numline_env import TIME, VALID_MOVES, NUM_SPACES, PantheonLine

STR_MOVES = [str(i) for i in VALID_MOVES]

def parse_obs(obs: np.ndarray) -> Tuple[int, int, int]:
    """Converts obs into ego-loc, alt-loc, time"""
    time = int(obs[-1])
    ego_loc = int(obs[time])
    alt_loc = int(obs[time + TIME])
    return ego_loc, alt_loc, time


def loc_to_string(ego_loc: int, alt_loc: int) -> str:
    """Convert observation into visual number-line"""
    print(ego_loc, alt_loc)
    lspace = -min(VALID_MOVES)
    rspace = max(VALID_MOVES)
    ego_loc += lspace
    alt_loc += lspace
    print_arr = [' '] * lspace + ['_'] * NUM_SPACES + [' '] * rspace
    if ego_loc == alt_loc:
        print_arr[ego_loc] = '*'
    else:
        print_arr[ego_loc] = 'E'
        print_arr[alt_loc] = 'A'
    return ''.join(print_arr)


class LineUser(Agent):
    """
    Agent representing human user input to the line environment
    """

    def __init__(self, env: PantheonLine):
        self.saw_tutorial = False
        self.env = env

    def print_tutorial(self):
        """Prints explanation of environment"""
        print("""
        In the Numberline environment, your goal is to match the
        location of the other player as closely as possible. Your
        reward is the negative sum of distances between your agent and
        the other agent after each of your moves.

        A perfect score is 0.0, while the worst score is -20.0, which
        is achieved by falling off of the number line on your first
        move, resulting in an instant end to the game.

        Your agent is denoted as E, and the other agent is denoted as
        A. If both of you are on the same space (the ideal), you will
        see a * instead of E or A. To move, just enter your movement,
        which can be -2, -1, 1, or 2. Note that you cannot stay
        'still' since a move of 0 is invalid.
        """)

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        if not self.saw_tutorial:
            self.print_tutorial()
            self.saw_tutorial = True
        obs = obs[0]  # remove mask
        ego_loc, alt_loc, time = parse_obs(obs)
        print(f"There are {time} timesteps remaining.")
        print("The current state is:")
        print(loc_to_string(ego_loc, alt_loc))
        move = -1
        while move == -1:
            action = input(f'Next move [{"/".join(STR_MOVES)}/(q)uit]: ').lower()
            if action in STR_MOVES:
                move = VALID_MOVES.index(int(action))
            elif action in ('q', 'quit'):
                print("Bye!")
                sys.exit(0)
            else:
                print("Invalid move!")
        return np.array([move])

    def update(self, reward: float, done: bool) -> None:
        if done:
            ego_loc, alt_loc = self.env.state
            print("Final state is:")
            print(loc_to_string(ego_loc, alt_loc))


class RandomLineAgent(Agent):
    """Agent representing random actor"""

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        return np.random.randint(0, len(VALID_MOVES), (1))

    def update(self, reward: float, done: bool) -> None:
        pass


class LeftBiasAgent(Agent):
    """Agent that is biased towards the left"""

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        obs = obs[0]
        ego, alt, _ = parse_obs(obs)
        diff = ego - alt
        if diff <= -4:
            move = 3
        elif diff == -3:
            move = 2
        elif diff == -2:
            move = 2
        elif diff == -1:
            if ego == 0:
                move = 3
            else:
                move = 1
        elif diff == 0:
            if ego == 0:
                move = 2
            else:
                move = 1
        elif diff == 1:
            if alt == 0:
                move = 2
            else:
                move = 0
        elif diff == 2:
            move = 1
        elif diff == 3:
            move = 0
        elif diff >= 4:
            move = 0
        return np.array([move])

    def update(self, reward: float, done: bool) -> None:
        pass


class RightBiasAgent(Agent):
    """Agent that is biased towards the right"""

    def get_action(self, obs: np.ndarray, record: bool = True) -> np.ndarray:
        obs = obs[0]
        ego, alt, _ = parse_obs(obs)
        diff = ego - alt
        if diff <= -4:
            move = 3
        elif diff == -3:
            move = 3
        elif diff == -2:
            move = 2
        elif diff == -1:
            if ego != NUM_SPACES - 1:
                move = 3
            else:
                move = 1
        elif diff == 0:
            if ego != NUM_SPACES - 1:
                move = 2
            else:
                move = 1
        elif diff == 1:
            if alt != NUM_SPACES - 1:
                move = 2
            else:
                move = 0
        elif diff == 2:
            move = 1
        elif diff == 3:
            move = 1
        elif diff >= 4:
            move = 0
        return np.array([move])

    def update(self, reward: float, done: bool) -> None:
        pass
