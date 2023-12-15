# %%
import gym
import numpy as np
import utils
import matplotlib.pyplot as plt
import pygame
from environments.inflinrewenvironment import InfGenInvestorGameEnv
from math import log
from functools import partial
import sys
# %%

def pow_util(pow, state, delta):
    if pow > 0:
        return (state + delta) ** pow - state ** pow
    else:
        return state ** pow - (state + delta) ** pow 
def log_util(state, delta):
    return log(1 + delta / state)

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
            phis = utils.extract_lin_rew_features(observation)
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
    models = ["sqrt", "cbrt", "lin", "log", "inv"]
    model = "lin"
    if len(sys.argv) >= 2:
        model = sys.argv[1]
        if model not in models:
            assert False
        elif len(sys.argv) >= 3:
            gamma = int(sys.argv[2])
        else:
            assert False
    gym.register(
        id='InfGenInvestorGameEnv-v0',
        entry_point='environments.inflinrewenvironment:InfGenInvestorGameEnv', 
        kwargs={'gamma': None, 'func': None} 
    )
    
    if model == "sqrt":
        env = gym.make('InfGenInvestorGameEnv-v0', gamma = 1/gamma, func = partial(pow_util, 1/2))
    if model == "cbrt":
        env = gym.make('InfGenInvestorGameEnv-v0', gamma = 1/gamma, func = partial(pow_util, 1/3))
    if model == "lin":
        env = gym.make('InfGenInvestorGameEnv-v0', gamma = 1/gamma, func = partial(pow_util, 1))
    if model == "log":
        env = gym.make('InfGenInvestorGameEnv-v0', gamma = 1/gamma, func = log_util)
    if model == "inv":
        env = gym.make('InfGenInvestorGameEnv-v0', gamma = 1/gamma, func = partial(pow_util, -1))
    np.random.seed(1234)
    theta, episode_rewards = train(env, N=1000, T=100, delta=1e-2)
    episode_rewards = np.array(episode_rewards)
    baseline = np.zeros_like(episode_rewards)
    f = open("params/inflinrew-" + model + ".txt", "w")
    for i in theta:
        f.write(str(i[0]))
        f.write(" ")
    f.close()
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
