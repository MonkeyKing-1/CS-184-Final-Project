# %%
import gym
import numpy as np
import utils
import matplotlib.pyplot as plt
import pygame
from environments.infgenenvironment import InfGenInvestorGameEnv
from rungen import log_util, pow_util
from math import log
from functools import partial
import sys

print_res = False

def plot_log_scores(name, scores):
    if name[:2] == "lb":
        name = "LB" + name[2:]
    else:
        name = name.title()
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

def act_lb(theta, state, lb):
    a = 2
    phis = utils.extract_lb_features(state, lb)
    dist = utils.compute_action_distribution(theta, phis)
    dist = np.reshape(dist, a)
    act = np.random.choice(np.arange(a), p=dist)
    return act

def infinite_gen(N, model, func):
    theta = np.loadtxt("params/infgen-" + model + ".txt")
    theta = theta.reshape((len(theta), 1))
    env = gym.make('InfGenInvestorGameEnv-v0', gamma = 0.1, func = func)
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
    print("Infinite " + model + " Utility Stats")
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    plot_log_scores(model + " util", scores)
    return scores

def infinite_lb(N, lb):
    theta = np.loadtxt("params/inflb-" + str(lb) + ".txt")
    theta = theta.reshape((len(theta), 1))
    env = gym.make('InfGenInvestorGameEnv-v0', gamma = 0.1, func = log_util)
    env.seed(1234)
    scores = []
    for i in range(N):
        state = env.reset()
        done = False
        while not done:
            pygame.event.get()
            action = act_lb(theta, state, lb)
            state, reward, done, _ = env.step(action)
        if print_res:
            print("Game " + str(i) + ":")
            print(state[0])
        scores.append(state[0])
    print("Infinite " + str(lb) + " LB Stats")
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    plot_log_scores("lb " + str(lb), scores)
    return scores

def piecewise_log_lin(N):
    theta_lin = np.loadtxt("params/infgen-lin.txt")
    theta_lin = theta_lin.reshape((len(theta_lin), 1))
    theta_log = np.loadtxt("params/infgen-log.txt")
    theta_log = theta_log.reshape((len(theta_log), 1))
    env = gym.make('InfGenInvestorGameEnv-v0', gamma = 0.1, func = partial(pow_util, 1))
    env.seed(12345)
    scores = []
    for i in range(N):
        state = env.reset()
        done = False
        while not done:
            pygame.event.get()
            if state[0] > 1800:
                action = act(theta_lin, state)
            else:
                action = act(theta_log, state)
            state, reward, done, _ = env.step(action)
        if print_res:
            print("Game " + str(i) + ":")
            print(state[0])
        scores.append(state[0])
    print("Infinite Piecewise Utility Stats")
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    plot_log_scores("Piecewise", scores)
    return scores

def cheese_strat(N, eps):
    env = gym.make('InfGenInvestorGameEnv-v0', gamma = 0.1, func = partial(pow_util, 1))
    env.seed(123456)
    scores = []
    for i in range(N):
        state = env.reset()
        done = False
        while not done:
            pygame.event.get()
            if state[1] > eps:
                action = 1
            else:
                action = 0
            state, reward, done, _ = env.step(action)
        if print_res:
            print("Game " + str(i) + ":")
            print(state[0])
        scores.append(state[0])
    print("Cheese Utility Stats")
    print("Average:", np.mean(scores))
    print("Std:", np.std(scores))
    plot_log_scores("Cheese", scores)
    return scores
    
# %%    
    
# if __name__ == '__main__':
util_models = [("sqrt", partial(pow_util, 1/2)), ("cbrt", partial(pow_util, 1/3)), ("lin", partial(pow_util, 1)), ("log", log_util), ("inv", partial(pow_util, -1))]
lb_models = [10, 100, 500]
num_runs = 20000
# model = "lin"
# num_runs = 10
# if len(sys.argv) >= 2:
#     model = sys.argv[1]
#     if model not in models:
#         assert False
#     elif len(sys.argv) >= 3:
#         num_runs = int(sys.argv[2])
#     if len(sys.argv) >= 4:
#         print_res = bool(sys.argv[3])
gym.register(
    id='InfGenInvestorGameEnv-v0',
    entry_point='environments.infgenenvironment:InfGenInvestorGameEnv', 
    kwargs={'gamma': None, 'func': None} 
)
# log_scores = infinite_gen(num_runs, model, log_util(state, delta))
scores = {}
for u in util_models:
    scores[u[0] + " util"] = infinite_gen(num_runs, u[0], u[1])
for l in lb_models:
    scores["lb " + str(l)] = infinite_lb(num_runs, l)
    
# %%    
winrates = np.zeros((8, 8))
models = list(scores.keys())
for i in range(8):
    for j in range(8):
        iscore = scores[models[i]]
        jscore = scores[models[j]]
        for k in range(num_runs):
            if iscore[k] > jscore[k]:
                winrates[i][j] += 1
                
winrates /= num_runs
winrates = winrates - np.transpose(winrates)
# %%        
print(winrates)
# %%
print(models)
# log_scores = infinite_gen(num_runs, "log", log_util)
# sqrt_scores = infinite_gen(num_runs, "sqrt", partial(pow_util, 1/2))
# cbrt_scores = infinite_gen(num_runs, "cbrt", partial(pow_util, 1/3))
# lin_scores = infinite_gen(num_runs, "lin", partial(pow_util, 1))
# inv_scores = infinite_gen(num_runs, "inv", partial(pow_util, -1))
# lb10_scores = infinite_lb(num_runs, 10)
# lb100_scores = infinite_lb(num_runs, 100)
# lb500_scores = infinite_lb(num_runs, 500)
# piecewise_scores = piecewise_log_lin(num_runs)
# cheese_scores = cheese_strat(num_runs, 0.9)
# other_scores = [sqrt_scores, cbrt_scores, lin_scores, inv_scores]
# other_scores = [piecewise_scores, cheese_scores]
# wins = [0] * len(other_scores)
# for j in range(len(other_scores)):
#     for i in range(num_runs):
#         if log_scores[i] > other_scores[j][i]:
#             wins[j] += 1
#     print(wins[j] / num_runs)
plt.show()

# %%

plt.xlim(4, 12)
plt.show()
# %%
for key in scores:
    plot_log_scores(key, scores[key])
plt.xlim(4, 10)
plt.show()
# %%
for key in scores:
    if key[:2] == "lb":
        plot_log_scores(key, scores[key])
plt.xlim(4, 10)
plt.show()
# %%
for key in scores:
    if key[:2] != "lb":
        plot_log_scores(key, scores[key])
plt.xlim(4, 10)
plt.show()
# %%
for key in scores:
    if key == "lb 500" or key == "log util" or key == "lin util":
        plot_log_scores(key, scores[key])
plt.xlim(4, 10)
plt.show()
# %%
for key in scores:
    kscores = scores[key]
    print(key)
    print("Average:", np.mean(kscores))
    print("Standard Deviation:", np.std(kscores))
    print("Median:", np.median(kscores))
    print("First Quartile:", np.quantile(kscores, 0.25))
    print("Third Quartile:", np.quantile(kscores, 0.75))
        
        
        
# %%
