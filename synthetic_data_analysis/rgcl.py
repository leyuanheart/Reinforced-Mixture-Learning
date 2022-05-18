# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:08:58 2022

@author: leyuan


reference:
    'A Reinforcement Learning Approach to Online Clustering.'  
                               --- Likas 1999 Neural Computation
"""

import numpy as np



from sklearn.utils import shuffle
from sklearn.metrics.pairwise import pairwise_distances_argmin_min, pairwise_distances


from scipy.stats import bernoulli


# ========== initialize W =================
def obj(X, W):    
    _, d = pairwise_distances_argmin_min(X, W)
    return d.sum()

def initial_W(X, K, seed=42):
    _, p = X.shape
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = np.random.default_rng(seed=seed)
    W_init = np.zeros((K, p))
    for j in range(p):
        W_init[:, j] = rng.uniform(mins[j], maxs[j], size=K)
        
    return W_init



def RGCL(X, K, alpha1=0.5, alpha2=0.1, max_step=1500, sus_exp=False, eta=1e-4, seed=42):
    '''
    reinforcement guided competitive learning.

    Parameters
    ----------
    X : numpy.array: (n, p)
        input data.
    K : int
        number of clusters.
    alpha1 : float, optional
        learning rate for the first 500 iteration. The default is 0.5.
    alpha2 : float, optional
        learning rate after the first 500 iteration. The default is 0.1.
    max_step : int, optional
        maximum number of iterations. The default is 1500.
    sus_exp : bool, optional
        sustained exploration. The default is False.
    eta : float, optional
        hyperparameter for sustained exploration. The default is 1e-5.
    seed : int, optional
        random seed. The default is 42.

    Returns
    -------
    J_list : list
        objective function value list.
    W : numpy.array: (K, p)
        output weight matrix.
    '''
    n, p = X.shape
    
    W_init = initial_W(X, K, seed=seed)
    W = np.copy(W_init)
    
    if sus_exp == False:
        for step in range(max_step):
            alpha = alpha1 if step < 500 else alpha2
            X_shuffle = shuffle(X, random_state=(step+seed))
            
            for i in range(n):
                x = X_shuffle[i, :]        
                
                # compute the probabilities
                s = pairwise_distances(np.array([x]), W)
                probs = 2 * (1 - 1 / (1 + np.exp(-s)))
                probs = np.ravel(probs)
                
                # output y
                y = bernoulli.rvs(probs)
                
                k_star =  np.argmax(probs)
                
                # compute the rewards
                rewards = np.zeros(K)
                rewards[k_star] = 1 if y[k_star] == 1 else -1
                
                # update W
                delta = alpha * rewards[k_star] * (y[k_star] - probs[k_star]) * (x - W[k_star, :])
                W[k_star, :] += delta                
        
        y_pred, _ = pairwise_distances_argmin_min(X, W)       
        return y_pred 
    
    else:
        J_min = 1e7
        for step in range(max_step):
            alpha = alpha1 if step < 500 else alpha2
            X_shuffle = shuffle(X, random_state=(step+seed))
            
            for i in range(n):
                x = X_shuffle[i, :]        
                
                # compute the probabilities
                s = pairwise_distances(np.array([x]), W)
                probs = 2 * (1 - 1 / (1 + np.exp(-s)))
                probs = np.ravel(probs)
                
                # output y
                y = bernoulli.rvs(probs)
                
                k_star =  np.argmax(probs)
                
                # compute the rewards
                rewards = np.zeros(K)
                rewards[k_star] = 1 if y[k_star] == 1 else -1
                
                # update W
                delta = alpha * rewards[k_star] * (y[k_star] - probs[k_star]) * (x - W[k_star, :])
                W[k_star, :] += delta - eta * W[k_star, :]
                W[np.arange(K) != k_star, :] -= eta * W[np.arange(K) != k_star, :]
                
            if step % 10 == 0:
                J = obj(X, W)
                if J < J_min:
                    J_min = J
                    W_min = W
        y_pred, _ = pairwise_distances_argmin_min(X, W_min)
        return y_pred















