from sklearn.kernel_approximation import RBFSampler
import numpy as np
from math import sqrt
from math import log

rbf_feature = RBFSampler(gamma=1, random_state=12345)

def extract_lin_rew_features(state):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s0 = np.zeros((3, 1))
    s1 = s0.copy()
    p = state[1]
    s1[0][0] = 1
    s1[1][0] = p * state[2] + (1 - p) * state[3]
    # s1[2][0] = (p * state[2] ** 2 + (1 - p) * state[3] ** 2) / state[0]
    # s1 /= state[0]
    s1[2][0] = p * log (state[2] / state[0] + 1) + (1 - p) * log (state[3] / state[0] + 1 + 10 ** -5)
    feats = np.concatenate((s0, s1), axis = -1)
    return feats

def extract_features(state):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s0 = np.zeros((3, 1))
    s1 = s0.copy()
    p = state[1]
    s1[0][0] = state[0]
    s1[1][0] = p * state[2] + (1 - p) * state[3]
    # s1[2][0] = (p * state[2] ** 2 + (1 - p) * state[3] ** 2) / state[0]
    s1 /= state[0]
    s1[2][0] = p * log (state[2] / state[0] + 1) + (1 - p) * log (state[3] / state[0] + 1 + 10 ** -5)
    feats = np.concatenate((s0, s1), axis = -1)
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    # TODO
    if logits[0][0] > logits[0][1] + 15:
        return np.array([[1, 0]])
    elif logits[0][1] > logits[0][0] + 15:
        return np.array([[0, 1]])
    pows = np.exp(logits)
    sums = np.sum(pows, axis = axis, keepdims=True)
    return pows/sums


def compute_action_distribution(theta, phis):
    """ compute probability distrubtion over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """

    logits = np.matmul(np.transpose(theta), phis)
    return compute_softmax(logits, 1)


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """
    d = phis.shape[0]
    logits = np.matmul(np.transpose(theta), phis) # 1 x |A|
    sfmx = compute_softmax(logits, 1)
    # apow = pows[0,action_idx]
    # powsum = np.sum(pows, axis=1)
    # powsum = powsum[0] # float
    phia = phis[:,action_idx]
    step = phia
    step = np.reshape(step, (-1, 1))
    step2 = np.matmul(phis, np.transpose(sfmx))
    grad = step - step2
    return grad
    # TODO
    pass


def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :param lamb: lambda value used for regularization 

    :return: fisher information matrix (shape d x d)
    
    

    Note: don't forget to take into account that trajectories might have different lengths
    """
    d = grads[0][0].shape[0]
    fisheracc = np.zeros((d, d))
    for i in range(len(grads)):
        fisher = np.zeros((d, d))
        for j in range(len(grads[i])):
            fisher = fisher + np.matmul(grads[i][j], np.transpose(grads[i][j]))
        fisher /= len(grads[i])
        fisheracc = fisheracc + fisher
    fisheracc /= len(grads)
    fisheracc += lamb * np.eye(d)
    return fisheracc
    

def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: list of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """

    # TODO
    b = 0
    d = grads[0][0].shape[0]
    N = len(rewards)
    reward = np.zeros(N)
    cumres = []
    ansgrad = np.zeros(d)
    for i in range(len(rewards)):
        cumre = []
        for j in range(len(rewards[i])):
            b += rewards[i][-j]
            reward[i] += rewards[i][-j]
            cumre.append(reward[i])
        cumre.reverse()
        cumres.append(cumre)
    b /= N
    for i in range(len(grads)):
        subgrad = np.zeros(d)
        for j in range(len(grads[i])):
            grad = np.reshape(grads[i][j], d)
            subgrad += grad * (cumres[i][j] - b)
        subgrad /= len(grads[i])
        ansgrad += subgrad
    ansgrad /= len(grads)
    ans = np.reshape(ansgrad, (d, 1))
    return ans
        

def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """

    # TODO
    eps = 10 ** -6
    denom = np.matmul(np.matmul(np.transpose(v_grad), np.linalg.inv(fisher)), v_grad)
    denom = denom[0][0] + eps
    eta = sqrt(delta/denom)
    return eta

