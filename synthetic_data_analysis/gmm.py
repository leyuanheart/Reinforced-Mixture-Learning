import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import random
import copy

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.datasets import make_blobs, make_classification
from sklearn.neighbors import DistanceMetric
from scipy.stats import multivariate_normal

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical

import multiprocessing as mp


def generate_data(n, p, K, pi, mu, sigma, seed=None):
    # if seed:
    #     np.random.seed(seed)
    # label = np.random.choice(range(K), size=(n, ), p=pi)
    # X = [np.random.multivariate_normal(mu[i], sigma[i]) for i in label]
    # X = np.array(X)
    if K:
        X, label, centers = make_blobs(n_samples=n, centers=K, n_features=p, cluster_std=sigma, return_centers=True, random_state=seed)
    if mu:
        X, label, centers = make_blobs(n_samples=n, centers=np.array(mu), n_features=p, cluster_std=sigma, return_centers=True, random_state=seed)
    
    return X, label, centers

# X, label = generate_data(n, p, K, pi, mu, sigma)



def get_data(x, batch_size=32):
    sample_size = x.shape[0]
    idx = np.random.choice(range(sample_size), batch_size, replace=False)
    return x[idx, :]


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, units=32):
        super(Actor, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(in_features=self.obs_dim, out_features=units)
        self.fc2 = nn.Linear(units, self.action_dim)
        
    def forward(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        logits = F.relu(self.fc1(obs))
        logits = self.fc2(logits)
        
        m = Categorical(logits=logits)
        action = m.sample()
        log_p = m.log_prob(action)
        
        return action, log_p, logits


# =======================================================================================
def metrics(label, y_pred):
    return adjusted_rand_score(label, y_pred), adjusted_mutual_info_score(label, y_pred)




def observed_loglikelihood(x, pi, mu, sigma):
    log_p = []
    for i in range(x.shape[0]):
        l_i = [p * multivariate_normal.pdf(x[i, :], m, s) for p, m, s in zip(pi, mu, sigma)]
        log_p.append(np.log(np.sum(l_i)))
    return log_p


def full_loglikelihood(x, pi, mu, sigma, label):
    log_p = []
    for i, l in enumerate(label):
        log_p.append(np.log(multivariate_normal.pdf(x[i, :], mu[l], sigma[l])) + np.log(pi[l]))
    return log_p



def compute_reward(x, K, actions, log_probs, dictionary, model_based=True):
    
    _, dim_x = x.shape
    reward_list = []
    mu_hat = []
    sigma_hat = []
    for k in range(K):
        # print((actions == k).sum())
        if (actions == k).sum() == 0:
            mu_hat.append(dictionary[k]['mu'])
        else:
            mu_hat.append(np.mean(x[actions == k, :], axis=0))
        
        if (actions == k).sum() <= dim_x+1:
            sigma_hat.append(dictionary[k]['sigma'])  
        else:
            sigma_hat.append(np.cov(x[actions==k, :], rowvar=False))
    
    pi_hat = [(actions == i).sum() / actions.size for i in range(K)]
        
    
    if model_based:
    ## model-based reward    
        reward_list = observed_loglikelihood(x, pi_hat, mu_hat, sigma_hat)
    else:
    ## model-free reward
        # intra-class distance
        s_w = 0
        for k in range(K):
            dist = DistanceMetric.get_metric('euclidean', V=sigma_hat[k])
            if (actions == k).sum() == 0:
                s_wk = 0
            else:
                s_wk = np.sum(dist.pairwise(x[actions==k,:], np.array([mu_hat[k]])))  
            # [(w - mu_hat[k]).dot(np.linalg.inv(sigma_hat[k])).dot((w - mu_hat[k]).T) for w in x[actions==k, :]]
            s_w += pi_hat[k] * s_wk
        # inter-class distance
        s_b = 0
        for l in range(K):
            for j in range(l, K):
                s_b += np.linalg.norm(mu_hat[l]-mu_hat[j])
        reward_list.append(s_b - s_w)
            
    return -np.array(reward_list), pi_hat, mu_hat, sigma_hat




## ========================= Simulation Setting 1 ======================================
n = 100  # 200, 300, 400
p = 2
K = 3
PI = [1/K for _ in range(K)] 
MU =  [[-3, 3], [0, 0], [3, 3]]    # [[-3] * p, [0] * p, [3] * p],  [[-1, -1], [0, 3], [3, 1]]
SIGMA = 1
COV = [np.diag([SIGMA] * p) for _ in range(K)]
model_based = True

units = 256
batch_size = 100
lr = 1e-3
max_steps = 5000

## ========================= Simulation Setting 2 ======================================
# n = 100   # 200, 300, 400
# p = 3
# K = 4
# PI = [1/K for _ in range(K)] 
# MU =  [[-8.5, -1.5, -1], [-1, 0, 8], [2.5, 7.5, -7], [3, -3, -0.5]]    # [[-3] * p, [0] * p, [3] * p]
# SIGMA = 2
# COV = [np.diag([SIGMA] * p) for _ in range(K)]

# units = 256
# batch_size = 100
# lr = 1e-3
# max_steps = 5000

## ========================= Simulation Setting 3 ======================================
# n = 100  # 200, 300, 400
# p = 10
# K = 3
# PI = [1/K for _ in range(K)] 
# MU =  [[-8, -9, 0, 8.5, 0, 5, 2, -7, 7.5, -4], 
#        [6, -2, 6, 3.5, 7, 9.5, -1.5, 4, -1.5, -2], 
#        [8.5, 9.5, 6, -4, 0, 4, -7.5, 5.5, 9, -3]]    
# SIGMA = 2.5
# COV = [np.diag([SIGMA] * p) for _ in range(K)]
# model_based = True

# units = 256
# batch_size = 100
# lr = 1e-3
# max_steps = 2500
## ========================= Simulation Settings ======================================



def params_estimate(pi, mu, sigma):
    order = np.argsort(mu, axis=0)[:,0]
    
    rmse1 = np.linalg.norm(np.array(pi)[order] - np.array(PI))
    rmse2 = np.sum([np.linalg.norm(mu1 - mu2) for mu1, mu2 in zip(np.array(mu)[order], np.array(MU))])
    rmse3 = np.sum([np.linalg.norm(sigma1 - sigma2) for sigma1, sigma2 in zip(np.array(sigma)[order], np.array(COV))])
    
    return rmse1, rmse2, rmse3

early_stop = True

def run(seed):  
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    X, label, _ = generate_data(n, p, K=None, pi=PI, mu=MU, sigma=SIGMA, seed=seed)     
    
    
    actor = Actor(obs_dim=p, action_dim=K, units=units)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    
    # scheduler=torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=500, gamma=0.9)
    # scheduler=optim.lr_scheduler.ReduceLROnPlateau(actor_optimizer, factor=0.9, patience=10)
        
    r_list = []
    r_baseline = torch.tensor(0)
    dictionary = dict()
    # for k in range(K):
    #     dictionary[k] = dict()
    #     dictionary[k]['mu'] =  np.zeros(p)
    #     dictionary[k]['sigma'] = np.diag([1.] * p)
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(X)
    for k in range(K):
        dictionary[k] = dict()
        dictionary[k]['mu'] =  np.mean(X[kmeans.labels_ == k, ], axis=0)
        dictionary[k]['sigma'] = np.cov(X[kmeans.labels_ == k, ], rowvar=False)
             
    
    for step in range(max_steps):
#         print('step: ', step)
        
        x_train = get_data(X, batch_size=batch_size)    
        
        actions, log_probs, _ = actor(x_train)
        
        
        # compute reward
        rewards, _, _, _ = compute_reward(x_train, K, actions.numpy(), log_probs, dictionary, model_based)
        r_list.append(rewards.mean())
#         print(f'average reward: {rewards.mean()}')
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        r_baseline = 0.95 * r_baseline + 0.05 * rewards.mean()
        
        # update actor
        actor_loss = ((rewards - r_baseline) * log_probs).mean()
        # actor_loss =  (rewards * log_probs).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()       # retain_graph=True if critic is used
        actor_optimizer.step()
        # print(f'actor loss: {actor_loss.item()}')
        
        # scheduler.step(actor_loss)
        
        if early_stop & (step > 10):
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
                & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3) \
                & (abs(r_list[-5] - r_list[-6]) < 1e-3) & (abs(r_list[-6] - r_list[-7]) < 1e-3) \
                & (abs(r_list[-7] - r_list[-8]) < 1e-3) & (abs(r_list[-8] - r_list[-9]) < 1e-3):
            
            # if abs(np.mean(r_list[:-10]) - np.mean(r_list[:-5])) < 1e-4:
                print(f'converge at step {step}')
                break
    
    with torch.no_grad():
        actions, log_probs, logits = actor(X)

    rewards, pi_hat, mu_hat, sigma_hat = compute_reward(X, K, actions.numpy(), log_probs, dictionary)
    
    gm = GaussianMixture(n_components=K, n_init=1, random_state=seed).fit(X)
    y_pred = gm.predict(X)
    
    
    results = np.zeros((2, 5))
    
    results[0, :2] = metrics(label, logits.argmax(dim=1))
    results[1, :2] = metrics(label, y_pred)
    
    results[0, 2:] = params_estimate(pi_hat, mu_hat, sigma_hat)
    results[1, 2:] = params_estimate(gm.weights_, gm.means_, gm.covariances_) 
    
    end = time.time()    
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    
    return results





if __name__ == '__main__':   
    # results = []
    # for sd in tqdm(range(50)):
    #     results.append(run(sd))

    # print("CPU的核数为：{}".format(mp.cpu_count()))
    start = time.time()
    pool = mp.Pool(10)
    dats = pool.map(run, range(100))
    pool.close()
    end = time.time()
    print(datetime.timedelta(seconds = end - start))
    
    
    dats = np.array([dat for dat in dats])

    np.save('n100.npy', dats)
    
    print(dats.mean(axis=0))
    print(dats.std(axis=0))