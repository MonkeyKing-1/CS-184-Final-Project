import gym
import pygame
from environments.finlinenvironment import FinLinInvestorGameEnv
from environments.inflinenvironment import InfLinInvestorGameEnv
from environments.finlogenvironment import FinLogInvestorGameEnv
from environments.inflogenvironment import InfLogInvestorGameEnv

# Register the environment
gym.register(
    id='InfLinInvestorGameEnv-v0',
    entry_point='environments.inflinenvironment:InfLinInvestorGameEnv', 
    kwargs={'gamma': None} 
)

# Test the environment
env = gym.make('InfLinInvestorGameEnv-v0', gamma = 0.1)
obs = env.reset()
env.render()

done = False
while not done:
    pygame.event.get()
    action = -1
    while(action not in [0, 1]):
        action = int(input("Invest?"))
    obs, reward, done, _ = env.step(action)
    env.render()
    print('Reward:', reward)
    print('Done:', done)

    pygame.time.wait(400)