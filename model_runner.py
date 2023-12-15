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

print_res = False

def plot_log_scores(name, scores):
    scores = np.log(scores)
    count, bins_count = np.histogram(scores, bins=100)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], cdf, label=name + " CDF")
    plt.legend()

def plot_scores(scores):
    count, bins_count = np.histogram(scores, bins=1000)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    plt.plot(bins_count[1:], pdf, label="PDF")
    plt.legend()

def act(theta, state):
    a = 2
    phis = utils.extract_features(state)
    dist = utils.compute_action_distribution(theta, phis)
    dist = np.reshape(dist, a)
    act = np.random.choice(np.arange(a), p=dist)
    return act

def infinite_linear(N):
    theta = np.loadtxt("params/infinite-linear_params.txt", delimiter=",")
    theta = theta.reshape((len(theta), 1))
    env = gym.make('InfLinInvestorGameEnv-v0', gamma = 0.1)
    env.seed(1234)
    scores = []
    for i in range(N):
        state = env.reset()
        done = False
        while not done:
            pygame.event.get()
            action = act(theta, state)
            state, reward, done, _ = env.step(action)
        if print_res:
            print("Game " + str(i) + ":")
            print(state[0])
        scores.append(state[0])
    print("Infinite Linear Utility Stats")
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    # plot_log_scores(scores)
    return scores

def infinite_log(N):
    theta = np.loadtxt("params/infinite-log_params.txt", delimiter=",")
    theta = theta.reshape((len(theta), 1))
    env = gym.make('InfLogInvestorGameEnv-v0', gamma = 0.1)
    env.seed(12345)
    scores = []
    for i in range(N):
        state = env.reset()
        done = False
        while not done:
            pygame.event.get()
            action = act(theta, state)
            state, reward, done, _ = env.step(action)
        if print_res:
            print("Game " + str(i) + ":")
            print(state[0])
        scores.append(state[0])
    print("Infinite Log Utility Stats")
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    # plot_log_scores(scores)
    return scores
    
def finite_linear(N):
    theta = np.loadtxt("params/finite-linear_params.txt", delimiter=",")
    theta = theta.reshape((len(theta), 1))
    env = gym.make('FinLinInvestorGameEnv-v0', gamma = 0.1)
    env.seed(123456)
    scores = []
    for i in range(N):
        state = env.reset()
        done = False
        while not done:
            pygame.event.get()
            action = act(theta, state)
            state, reward, done, _ = env.step(action)
        if print_res:
            print("Game " + str(i) + ":")
            print(state[0])
        scores.append(state[0])
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    return scores
    
def finite_log(N):
    theta = np.loadtxt("params/finite-log_params.txt", delimiter=",")
    theta = theta.reshape((len(theta), 1))
    env = gym.make('FinLogInvestorGameEnv-v0', gamma = 0.1)
    env.seed(1234567)
    scores = []
    for i in range(N):
        state = env.reset()
        done = False
        while not done:
            pygame.event.get()
            action = act(theta, state)
            state, reward, done, _ = env.step(action)
        if print_res:
            print("Game " + str(i) + ":")
            print(state[0])
        scores.append(state[0])
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    return scores
    
    
    
if __name__ == '__main__':
    models = ["infinite-linear", "finite-linear", "infinite-log", "finite-log"]
    model = "infinite-linear"
    num_runs = 10
    if len(sys.argv) >= 2:
        model = sys.argv[1]
        if model not in models:
            assert False
        elif len(sys.argv) >= 3:
            num_runs = int(sys.argv[2])
        if len(sys.argv) >= 4:
            print_res = bool(sys.argv[3])
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
        infinite_linear(num_runs)
    if model == "finite-linear":
        finite_linear(num_runs)
    if model == "infinite-log":
        inflog = infinite_log(num_runs)
        inflin = infinite_linear(num_runs)
        plot_log_scores("infinite log utility", inflog)
        plot_log_scores("infinite linear utility", inflin)
        print(np.min(inflog))
        plt.show()
        logwins = 0
        basewins = 0
        for i in range(num_runs):
            if inflog[i] > inflin[i]:
                logwins += 1
            if inflog[i] > 1000:
                basewins += 1
        print(logwins / num_runs)
    if model == "finite-log":
        finite_log(num_runs)
    