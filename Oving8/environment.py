import numpy as np
import pygame
from time import sleep

def drawGrid(screen, color, goal, user, q_table):
    blockSize = 100
    use_color = color
    scalar = 2.5 if np.max(q_table) < 2 else 1
    for x in range(0, 700, blockSize):
        for y in range(0, 700, blockSize):
            use_color = color

            rect = pygame.Rect(x, y, blockSize, blockSize)

            q_val = (np.max(q_table[(int(x / blockSize), int(y / blockSize))]) + 1) * scalar
            if q_val > 0:
                temp = np.array(use_color)
                temp[0] = 255
                temp[1] = 255 - int(20 * q_val)
                temp[2] = 0
                #print(tuple(temp))
                pygame.draw.rect(screen, tuple(temp), rect)

            if (x, y) == tuple(np.multiply(goal, blockSize)):
                pygame.draw.rect(screen, (0, 255, 0), rect, 60, 60)
            if (x, y) == tuple(np.multiply(user, blockSize)):
                pygame.draw.rect(screen, (0, 0, 255), rect, 60, 60)

            pygame.draw.rect(screen, use_color, rect, 1)

class Environment:

    def __init__(self, x = 7):
        self.agent = None
        self.x = x
        # Creates two random integers from 5-9
        self.start_pos = np.random.randint(x-2, x-1, size=2)
        # Grid 5x5
        self.rewards = np.zeros((x, x))
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
        if action == 1: return self.agent[0] < (self.x-1)
        if action == 2: return self.agent[1] > 0
        if action == 3: return self.agent[1] < (self.x-1)
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

        #print(self.rewards)

    def manhatten_distance(self, node):
        return abs(self.goal[0] - node[0]) + abs(self.goal[1] - node[1])

    def render(self, q_table):
        pygame.init()

        screen = pygame.display.set_mode([700, 700])
        pygame.display.set_caption("GridGame")

        black = (0, 0, 0)
        white = (255, 255, 255)

        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True

            state = self.reset()

            done1 = False
            while not done1:
                action = np.argmax(q_table[state])
                state, _, done1 = self.step(action)
                sleep(0.1)
                screen.fill(black)
                drawGrid(screen, white, self.goal, self.agent, q_table)
                pygame.display.flip()

        pygame.quit()