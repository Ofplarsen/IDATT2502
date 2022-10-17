import math
from abc import ABC
import numpy as np

import gym


class Environment:

    def __init__(self):
        self.agent = None
        # Creates two random integers from 5-9
        self.start_pos = np.random.randint(5, 9, size=2)
        # Grid 5x5
        self.rewards = np.zeros((10, 10))
        # Goal is in (0,0)
        self.goal = np.array([0, 0])
        self.reset()
        self.init_rewards()


    # One of two needed for env
    def reset(self):
        # Sets agent/player to start pos
        self.agent = np.array(self.start_pos)
        # Returns start post as tuple
        return tuple(self.agent)

    # ACTIONS!
    # 0: Left, 1: Right, 2: Up, 3: Down

    def valid_move(self, action):
        # Checks if action results in agent outside of grid
        if action == 0: return self.agent[0] > 0
        if action == 1: return self.agent[0] < 9
        if action == 2: return self.agent[1] > 0
        if action == 3: return self.agent[1] < 9
        return False

    # Creates new state based on action, etc if move one right, get that square etc
    def next_state(self, action):
        if action == 0: self.agent[0] -= 1
        if action == 1: self.agent[0] += 1
        if action == 2: self.agent[1] -= 1
        if action == 3: self.agent[1] += 1

    # Used to make an action / take a step

    def step(self, action):
        done = tuple(self.agent) == tuple(self.goal)
        reward = -1
        if self.valid_move(action):
            self.next_state(action)
            reward = self.rewards[tuple(self.agent)]

        # Returns state, reward, done
        return tuple(self.agent), reward, done

    # Fills rewards using manhatten distance (finding distrance when only allowed to move by squares)
    def init_rewards(self):
        for x in range(len(self.rewards)):
            for y in range(len(self.rewards[0])):
                self.rewards[x, y] = 1 - self.manhatten_distance([x, y]) ** 0.4

        print(self.rewards)

    def manhatten_distance(self, node):
        return abs(self.goal[0] - node[0]) + abs(self.goal[1] - node[1])
