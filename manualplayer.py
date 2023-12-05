import gym
from environment import InvestorGameEnv
import pygame

# Register the environment
gym.register(
    id='InvestorGameEnv-v0',
    entry_point='environment:InvestorGameEnv', 
    kwargs={'gamma': None} 
)

# Test the environment
env = gym.make('InvestorGameEnv-v0',max_rounds = 20)
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