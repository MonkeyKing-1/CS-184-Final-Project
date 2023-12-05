import gym
from gym import spaces
import numpy as np
import pygame
from math import log


black = (0, 0, 0)

def get_next_investment(position):
    # Most investments are likely to succeed(?)
    prob = np.random.beta(1, 1)
    downside = -position * np.random.beta(3, 2) * 0.5
    upside = -downside * (1 - prob) / prob * (0.5 + 1.5 * np.random.uniform())
    return (position, prob, upside, downside)

class InfLinInvestorGameEnv(gym.Env):
    def __init__(self, gamma):
        super(InfLinInvestorGameEnv, self).__init__()
        self.gamma = gamma
        self.start_pos = 1000.0
        self.cur_pos = self.start_pos
        _, self.prob, self.upside, self.downside = get_next_investment(self.cur_pos)
        # 4 possible actions: 0=dont invest, 1=invest
        self.action_space = spaces.Discrete(2)  
        self.round = 0
        self.max_rounds = np.random.geometric(p = self.gamma)

        # Observation space is current position, probability of successful investment, 
        # successful investment payout, and failure payout
        self.observation_space = spaces.Box(low=np.array([float('-inf'), float('-inf'), 0.0, float('-inf')]), high=np.array([float('inf'), float('inf'), 1.0, float('inf')]), dtype=float)

        # Initialize Pygame
        pygame.init()

        # setting display size
        self.screen = pygame.display.set_mode((400, 400))

    def reset(self):
        self.cur_pos = self.start_pos
        self.round = 0
        self.max_rounds = np.random.geometric(p = self.gamma)
        _, self.prob, self.upside, self.downside = get_next_investment(self.cur_pos)
        return self.wrap_outputs()

    def step(self, action):
        # Move the agent based on the selected action
        if action == 1:
            samp = np.random.uniform()
            if samp > self.prob:
                reward = self.downside
            else:
                reward = self.upside
        else:
            reward = 0
        self.cur_pos += reward
        self.round += 1
        # print(self.round, self.max_rounds)
        # Reward function
        if self.round == self.max_rounds or self.cur_pos < 10.00:
            done = True
        else:
            done = False
        _, self.prob, self.upside, self.downside = get_next_investment(self.cur_pos)
        return self.wrap_outputs(), reward, done, {}

    def render(self, mode = None):
        # Clear the screen
        self.screen.fill((255, 255, 255))  

        # Draw env elements one cell at a time
        font = pygame.font.Font('freesansbold.ttf', 32)
 
        # create a text surface object,
        # on which text is drawn on it.
        text1 = font.render("Position: " + "%.2f" % self.cur_pos, True, 100)
        # text2 = font.render("Success Prob: " + "%.2f" % self.prob, True, 100)
        text3 = font.render("Upside: " + "%.2f" % self.upside, True, 100)
        text4 = font.render("Downside: " + "%.2f" % self.downside, True, 100)
        texts = [text1, text3, text4]
        
        for i in range(0, 3):
            textRect = texts[i].get_rect()
            textRect.center = (200, 50 + 100 * i)
            self.screen.blit(texts[i], textRect)
        
        pygame.display.update()  # Update the display
    def wrap_outputs(self):
        return np.array([self.cur_pos, self.prob, self.upside, self.downside])