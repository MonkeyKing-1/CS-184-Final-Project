# %%
import gym
import numpy as np
import utils
import matplotlib.pyplot as plt
import pygame
from environments.finlinenvironment import FinLinInvestorGameEnv
from environments.inflinenvironment import InfLinInvestorGameEnv
from environments.finlogenvironment import FinLogInvestorGameEnv
from environments.inflogenvironment import InfLogInvestorGameEnv
from math import log
import sys
# %%

gamma = 20

def sample(theta, env : gym.Env, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []
    d = theta.shape[0]
    a = 2
    for i in range(N):
        observation = env.reset()
        rewards = []
        grads = []
        len = 0
        while True:
            phis = utils.extract_features(observation)
            dist = utils.compute_action_distribution(theta, phis)
            dist = np.reshape(dist, a)
            act = np.random.choice(np.arange(a), p=dist)
            observation, reward, done, _ = env.step(act)
            rewards.append(reward)
            grad = utils.compute_log_softmax_grad(theta, phis, act)
            grads.append(grad)
            if done:
                break
        total_rewards.append(rewards)
        total_grads.append(grads)
    
    # TODO
    
    

    return total_grads, total_rewards

def train(env, N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    # theta = np.random.rand(4, 1)
    # theta -= 0.5
    theta = np.random.rand(3, 1)

    episode_rewards = []
    for i in range(T):
        print(i)
        grads, rewards = sample(theta, env, N)
        rsum = 0
        for reward in rewards:
            # rsubsum = 1000
            rsubsum = 0
            for r in reward:
                rsubsum += r
            # rsum += log(rsubsum) - log(1000)
            rsum += rsubsum
        rsum /= N
        fisher = utils.compute_fisher_matrix(grads)
        v_grad = utils.compute_value_gradient(grads, rewards)
        eta = utils.compute_eta(delta, fisher, v_grad)
        theta += eta * np.matmul(np.linalg.inv(fisher), v_grad)
        print(theta)
        episode_rewards.append(rsum)

    return theta, episode_rewards

if __name__ == '__main__':
    models = ["infinite-linear", "finite-linear", "infinite-log", "finite-log"]
    model = "infinite-linear"
    if len(sys.argv) >= 2:
        model = sys.argv[1]
        if model not in models:
            assert False
        elif len(sys.argv) >= 3:
            gamma = int(sys.argv[2])
        else:
            assert False
    gym.register(
        id='InfLinInvestorGameEnv-v0',
        entry_point='environments.inflinenvironment:InfLinInvestorGameEnv', 
        kwargs={'gamma': None} 
    )
    gym.register(
        id='FinLinInvestorGameEnv-v0',
        entry_point='environments.finlinenvironment:FinLinInvestorGameEnv', 
        kwargs={'max_rounds': None} 
    )
    gym.register(
        id='InfLogInvestorGameEnv-v0',
        entry_point='environments.inflogenvironment:InfLogInvestorGameEnv', 
        kwargs={'gamma': None} 
    )
    gym.register(
        id='FinLogInvestorGameEnv-v0',
        entry_point='environments.finlogenvironment:FinLogInvestorGameEnv', 
        kwargs={'max_rounds': None} 
    )
    
    if model == "infinite-linear":
        env = gym.make('InfLinInvestorGameEnv-v0', gamma = 1/gamma)
    if model == "finite-linear":
        env = gym.make('FinLinInvestorGameEnv-v0', max_rounds = gamma)
    if model == "infinite-log":
        env = gym.make('InfLogInvestorGameEnv-v0', gamma = 1/gamma)
    if model == "finite-log":
        env = gym.make('FinLogInvestorGameEnv-v0', max_rounds = gamma)
    np.random.seed(1234)
    theta, episode_rewards = train(env, N=1000, T=100, delta=1e-2)
    episode_rewards = np.array(episode_rewards)
    baseline = np.zeros_like(episode_rewards)
    if model == "infinite-linear" or model == "finite-linear":
        episode_rewards = np.log(1000 + episode_rewards)
        baseline = np.ones_like(episode_rewards) * log(1000)
    plt.plot(episode_rewards)
    plt.plot(baseline)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()
    # %%
    def act(theta, state):
        a = 2
        phis = utils.extract_features(state)
        dist = utils.compute_action_distribution(theta, phis)
        # print(dist)
        dist = np.reshape(dist, a)
        act = np.random.choice(np.arange(a), p=dist)
        return act
    env.seed(12345)
    state = env.reset()
    done = False
    print(theta)
    while not done:
        pygame.event.get()
        action = act(theta, state)
        state, reward, done, _ = env.step(action)
        env.render()
        print('Reward:', reward)
        print('Done:', done)
        print(state[0])
        pygame.time.wait(400)

# %%
