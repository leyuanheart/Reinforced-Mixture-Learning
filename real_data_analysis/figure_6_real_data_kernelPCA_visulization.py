# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:38:53 2022

"""

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
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.manifold import spectral_embedding
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.utils import shuffle
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA

from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.stats import multivariate_normal

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.distributions import Categorical

import os

from rgcl import RGCL


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



def compute_reward(x, K, actions, log_probs, dictionary, model_based=True, simplified=False):
    _, dim_x = x.shape
    reward_list = []
    mu_hat = []
    sigma_hat = []
    for k in range(K):
        # print((actions == k).sum())
        if (actions == k).sum() == 0:
            # print('mu can not be calculated')
            mu_hat.append(dictionary[k]['mu'])
        else:
            mu_hat.append(np.mean(x[actions == k, :], axis=0))
        
        if (actions == k).sum() <= dim_x+1:    # 变量维数
            # print('cov can not be calculated')
            sigma_hat.append(dictionary[k]['sigma'])  
        else:
            sigma_hat.append(np.cov(x[actions==k, :], rowvar=False))
    
    pi_hat = [(actions == i).sum() / actions.size for i in range(K)]
        
    # for i, action in enumerate(actions):
        
        # reward = multivariate_normal.pdf(x[i, :], mu_hat[action], sigma_hat[action])  # 这里用的是conditional likelihood, 是不是考虑用full likelihood, 即还要乘上每个component的概率pi_hat[k]
        # reward_list.append(- np.log(reward))
        
        # reward = multivariate_normal.pdf(x[i, :], mu_hat[action], sigma_hat[action])  
        # reward_list.append(- np.log(reward) - np.log(pi_hat[action]))
        
        
        # dist = np.linalg.norm(x[i, :] - mu_hat[action])
        # reward_list.append(dist)
        
        
        # cosine distance does not work
        # cosine = np.dot(x[i, :], mu_hat[action]) / (np.linalg.norm(x[i, :]) * np.linalg.norm(mu_hat[action]))
        # reward_list.append(cosine)
    
        
    if model_based:
    ## model-based reward    
        # for i in range(actions.size):
            # r = [pi * multivariate_normal.pdf(x[i, :], mu, sigma) for pi, mu, sigma in zip(pi_hat, mu_hat, sigma_hat)]
            # reward_list.append(-np.log(np.sum(r)))
        reward_list = observed_loglikelihood(x, pi_hat, mu_hat, sigma_hat)
        # reward_list = full_loglikelihood(x, pi_hat, mu_hat, sigma_hat, actions)
    else:
    ## model-free reward
        if simplified:
            for i, action in enumerate(actions):
                dist = np.linalg.norm(x[i, :] - mu_hat[action])
                reward_list.append(-dist)                              # 简单的类内散度有时比复杂的类间散度-内类散度要好
            
            return -np.array(reward_list), pi_hat, mu_hat, sigma_hat
            
        # intra-class distance
        s_w = 0
        for k in range(K):
            dist = DistanceMetric.get_metric('euclidean')
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
        
        for _ in range(actions.size):
            reward_list.append(s_b - s_w)   # 减的效果比除要好
        
            
    return -np.array(reward_list), pi_hat, mu_hat, sigma_hat


def metrics(label, y_pred):
    return adjusted_rand_score(label, y_pred), adjusted_mutual_info_score(label, y_pred), homogeneity_score(label, y_pred), completeness_score(label, y_pred)

def observed_loglikelihood(x, pi, mu, sigma):
    log_p = []
    for i in range(x.shape[0]):
        l_i = [p * multivariate_normal.pdf(x[i, :], m, s) for p, m, s in zip(pi, mu, sigma)]
        log_p.append(np.log(np.sum(l_i)))
    return log_p


def spectral_embeddings(X, graph, mode, n_neighbors, k, normed=True):
    '''
    graph: str, 'k_nearest' or 'kernel_based'
    mode: str, 'distance' or 'connectivity'
    n_neighbors: int, only available when graph == 'k_nearest'
    k: int, num of eigen vectors to choose
    '''
    n, p = X.shape
    
    # =================== compute the similarity matrix (graph) ==========================
    if graph == 'k_nearest':
        knn_dist_graph = kneighbors_graph(  X=X,
                                            n_neighbors=n_neighbors,
                                            mode=mode,
                                            include_self=True,
                                            metric='euclidean')                
        if mode == 'distance':
            gamma = 1 / p
            similarity_graph = sparse.lil_matrix(knn_dist_graph.shape)
            nonzeroindices = knn_dist_graph.nonzero()
            
            similarity_graph[nonzeroindices] = np.exp(-np.square(knn_dist_graph[nonzeroindices]) * gamma)            
            similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
        else:
            similarity_graph = 0.5 * (knn_dist_graph + knn_dist_graph.T)
    elif graph ==  'kernel_based':
        similarity_graph = pairwise_kernels(X, metric='rbf')    
        similarity_graph = 0.5 * (similarity_graph + similarity_graph.T)
    else:
        raise TypeError("mode must be 'k_nearest' or 'kernel_based'.")
        
    
    # ================================ compute the laplacian of the graph ============
    L, degree = csgraph_laplacian(similarity_graph, normed=normed, return_diag=True)
    
    # ========================== eigen decompostion of the laplacian =================
    if isinstance(L, sparse.coo.coo_matrix):
        L = L.todense()
    e, evecs = eigh(L)
    
    embeddings = evecs[:, :k]
    
    return embeddings




def figure_6():
    pass
# ========================================================================
# fig, axes = plt.subplots(3, 6)
# plt.subplots_adjust(top=0.935,
#                     bottom=0.067,
#                     left=0.035,
#                     right=0.992,
#                     hspace=0.273,
#                     wspace=0.269)
plt.style.use('seaborn-paper')

model_based = False
early_stop = False
units = 256
lr = 1e-3


def Iris(seed, batch_size, max_steps = 5000): 
    '''
    reward: s_b - s_w
    without spectral embedding
    '''
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    data = datasets.load_iris()
    X, label = shuffle(data.data, data.target, random_state=seed)
    
    n, p = X.shape
    K = len(np.unique(label))
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
        
        x_train = get_data(X, batch_size=batch_size)    # batch size小的话，training step就要大一些
        
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
        
        if early_stop & (step > 6):
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
                & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
            
            # if abs(np.mean(r_list[:-10]) - np.mean(r_list[:-5])) < 1e-4:
                print(f'converge at step {step}')
                break    
    
    with torch.no_grad():
        actions, log_probs, logits = actor(X)

#     acp_RI, acp_MI, acp_HS, acp_CS  = metrics(label, logits.argmax(dim=1))
    
    results = np.zeros((7, 4))
    
    results[0, :] = metrics(label, logits.argmax(dim=1))
    
    gm = GaussianMixture(n_components=K, n_init=1, random_state=seed).fit(X)
    gm_pred = gm.predict(X)
    results[1, :] = metrics(label, gm_pred)
    
    km = KMeans(n_clusters=K, random_state=seed).fit(X)
    km_pred = km.predict(X)
    results[2, :] = metrics(label, km_pred)
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    sc_pred = sc.labels_
    results[3, :] = metrics(label, sc_pred)
       
    
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    al_pred = average_linkage.labels_
    results[4, :] = metrics(label, al_pred)
    
    rgcl_pred = RGCL(X, K, seed=seed)
    results[5, :] = metrics(label, rgcl_pred)
    
    
    srgcl_pred = RGCL(X, K, sus_exp=True, seed=seed)
    results[6, :] = metrics(label, srgcl_pred)
    
    # dbscan = cluster.DBSCAN(eps=0.5, min_samples=5).fit(X)
    # y_pred = dbscan.labels_
    # results[5, :] = metrics(label, y_pred)
    
    # optics = cluster.OPTICS(min_samples=5, xi=0.05).fit(X)
    # y_pred = optics.labels_
    # results[6, :] = metrics(label, y_pred)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results, X, label, logits.argmax(dim=1).numpy(), gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred


iris_results, iris, label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred = Iris(seed=20, batch_size=150)

label = np.where(label==0, -1, label)
label = np.where(label==1, 0, label)
label = np.where(label==-1, 1, label)

acp_pred = acp_pred + 1
acp_pred = np.where(acp_pred==3, 0, acp_pred)

rgcl_pred = rgcl_pred + 1
rgcl_pred = np.where(rgcl_pred==3, 0, rgcl_pred)

srgcl_pred = srgcl_pred + 1
srgcl_pred = np.where(srgcl_pred==3, 0, srgcl_pred)


# par = range(60, 90)
# fig, axes = plt.subplots(5, 6)
# idx = 0
# for i in range(5):
#     for j in range(6):
#         X_embedded = TSNE(n_components=2, perplexity=par[idx], random_state=42).fit_transform(X)    
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)
#         idx += 1

# X_embedded = TSNE(n_components=2, perplexity=64, random_state=42).fit_transform(X) 
data = datasets.load_iris()
iris, label = shuffle(data.data, data.target, random_state=20)
X_embedded = KernelPCA(n_components=2, kernel='rbf').fit_transform(iris)
# plt.figure()
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=label)



iris_label_list = [label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred]
names = ['Ground Truth', 'ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering', 'RGCL', 'SRGCL']

metric_names = ['Adjusted\nRand Index', 
                'Adjusted\nMutual Information', 
                'Homogeneity', 
                'Completeness']


# fig, axes = plt.subplots(2, 3)
# plt.subplots_adjust(top=0.955,
#                     bottom=0.1,
#                     left=0.045,
#                     right=0.985,
#                     hspace=0.35,
#                     wspace=0.2)
# idx = 0
# for i in range(2):
#     for j in range(3):
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], 
#                            c=label_list[idx], alpha=1)
#         axes[i, j].set_title(names[idx], fontdict={'fontsize': 20})
#         if i !=0 or j != 0:
#             axes[i, j].set_xlabel(f'Adjusted Rand Index: {results[idx-1, 0]:.2f}, Adjusted Mutual Information: {results[idx-1, 1]:.2f}\nHomogeneity: {results[idx-1, 2]:.2f}, Completeness: {results[idx-1, 3]:.2f}')
#         idx += 1
# axes[0, 0].set_xlabel('Iris Plants', fontdict={'fontsize': 15})
# # fig.suptitle('Iris Plants')

fig, axes = plt.subplots(6, 8)
# plt.subplots_adjust(top=0.950,
#                     bottom=0.067,
#                     left=0.035,
#                     right=0.992,
#                     hspace=0.273,
#                     wspace=0.269)
plt.style.use('seaborn-darkgrid')
idx = 0
for j in range(8):
    axes[0, j].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                        c=iris_label_list[idx], alpha=1, s=14)
    axes[0, j].set_title(names[idx], fontdict={'fontsize': 14})
    if j != 0:
        axes[0, j].set_xlabel(f'Adjusted RI: {iris_results[idx-1, 0]:.2f}, Adjusted MI: {iris_results[idx-1, 1]:.2f}\nHomogeneity: {iris_results[idx-1, 2]:.2f}, Completeness: {iris_results[idx-1, 3]:.2f}', fontsize=8)
        axes[0, j].set_xticks(())
        axes[0, j].set_yticks(())
    idx += 1
axes[0, 0].set_xlabel('Iris Plants', fontdict={'fontsize': 15})
axes[0, 5].set_title(names[5], fontdict={'fontsize': 13})
# ========================================================================



graph = 'k_nearest'  # k_nearest, kernel_based
mode = 'connectivity'    # distance, connectivity
n_neighbors = 20


def Wine(seed, batch_size, max_steps = 5000): 
    '''
    reward: s_b
    with spectral embedding
    '''
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    data = datasets.load_wine()
    X, label = shuffle(data.data, data.target, random_state=seed)
    K = len(np.unique(label))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    embeddings = spectral_embeddings(X, graph, mode, n_neighbors, K)
        
    
    actor = Actor(obs_dim=K, action_dim=K, units=units)
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
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(embeddings)
    for k in range(K):
        dictionary[k] = dict()
        dictionary[k]['mu'] =  np.mean(embeddings[kmeans.labels_ == k, ], axis=0)
        dictionary[k]['sigma'] = np.cov(embeddings[kmeans.labels_ == k, ], rowvar=False)
             
    
    for step in range(max_steps):
#         print('step: ', step)
        
        x_train = get_data(embeddings, batch_size=batch_size)    # batch size小的话，training step就要大一些
        
        actions, log_probs, _ = actor(x_train)
        
        
        # compute reward
        rewards, _, _, _ = compute_reward(x_train, K, actions.numpy(), log_probs, dictionary, model_based, True)
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
        
        if early_stop & (step > 6):
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
                & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
            
            # if abs(np.mean(r_list[:-10]) - np.mean(r_list[:-5])) < 1e-4:
                print(f'converge at step {step}')
                break
    
    
    with torch.no_grad():
        actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

#     acp_RI, acp_MI, acp_HS, acp_CS  = metrics(label, logits.argmax(dim=1))
    
    results = np.zeros((7, 4))
    
    results[0, :] = metrics(label, logits.argmax(dim=1))
    
    gm = GaussianMixture(n_components=K, n_init=1, random_state=seed).fit(X)
    gm_pred = gm.predict(X)
    results[1, :] = metrics(label, gm_pred)
    
    km = KMeans(n_clusters=K, random_state=seed).fit(X)
    km_pred = km.predict(X)
    results[2, :] = metrics(label, km_pred)
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    sc_pred = sc.labels_
    results[3, :] = metrics(label, sc_pred)
    
    
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    al_pred = average_linkage.labels_
    results[4, :] = metrics(label, al_pred)
    
    rgcl_pred = RGCL(X, K, seed=42)
    results[5, :] = metrics(label, rgcl_pred)
    
    
    srgcl_pred = RGCL(X, K, sus_exp=True, seed=seed)
    results[6, :] = metrics(label, srgcl_pred)
    
    # dbscan = cluster.DBSCAN(eps=0.5, min_samples=5).fit(X)
    # y_pred = dbscan.labels_
    # results[5, :] = metrics(label, y_pred)
    
    # optics = cluster.OPTICS(min_samples=5, xi=0.05).fit(X)
    # y_pred = optics.labels_
    # results[6, :] = metrics(label, y_pred)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results, X, label, logits.argmax(dim=1).numpy(), gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred



wine_results, wine, label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred = Wine(seed=5, batch_size=100)


label = np.where(label==0, -1, label)
label = np.where(label==1, 0, label)
label = np.where(label==-1, 1, label)

acp_pred = acp_pred + 1
acp_pred = np.where(acp_pred==3, 0, acp_pred)

km_pred = km_pred + 1
km_pred = np.where(km_pred==1, 0, km_pred)
km_pred = np.where(km_pred==3, 1, km_pred)

al_pred = np.where(al_pred==0, -1, al_pred)
al_pred = np.where(al_pred==1, 0, al_pred)
al_pred = np.where(al_pred==-1, 1, al_pred)

srgcl_pred += 1
srgcl_pred = np.where(srgcl_pred==3, 0, srgcl_pred)



# par = range(30, 60)
# fig, axes = plt.subplots(5, 6)
# idx = 0
# for i in range(5):
#     for j in range(6):
#         X_embedded = TSNE(n_components=2, perplexity=par[idx], random_state=42).fit_transform(X)    
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)
#         idx += 1


# X_embedded = TSNE(n_components=2, perplexity=31, random_state=42).fit_transform(X)
data = datasets.load_wine()
X, label = shuffle(data.data, data.target, random_state=5)
scaler = StandardScaler()
wine = scaler.fit_transform(X)
X_embedded = KernelPCA(n_components=2, kernel='rbf').fit_transform(wine)
# plt.figure()
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=label)




wine_label_list = [label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred]
# names = ['Ground Truth', 'ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering']

# fig, axes = plt.subplots(2, 3)
# plt.subplots_adjust(top=0.955,
#                     bottom=0.1,
#                     left=0.045,
#                     right=0.985,
#                     hspace=0.35,
#                     wspace=0.2)
# idx = 0
# for i in range(2):
#     for j in range(3):
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label_list[idx])
#         axes[i, j].set_title(names[idx], fontdict={'fontsize': 20})
#         if i !=0 or j != 0:
#             axes[i, j].set_xlabel(f'Adjusted Rand Index: {results[idx-1, 0]:.2f}, Adjusted Mutual Information: {results[idx-1, 1]:.2f}\nHomogeneity: {results[idx-1, 2]:.2f}, Completeness: {results[idx-1, 3]:.2f}')
#         idx += 1
# axes[0, 0].set_xlabel('Wine recognition', fontdict={'fontsize': 15})


idx = 0
for j in range(8):
    axes[1, j].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                        c=wine_label_list[idx], alpha=1, s=14)
    # axes[1, j].set_title(names[idx], fontdict={'fontsize': 15})
    if j != 0:
        axes[1, j].set_xlabel(f'Adjusted RI: {wine_results[idx-1, 0]:.2f}, Adjusted MI: {wine_results[idx-1, 1]:.2f}\nHomogeneity: {wine_results[idx-1, 2]:.2f}, Completeness: {wine_results[idx-1, 3]:.2f}', fontsize=8)
        axes[1, j].set_xticks(())
        axes[1, j].set_yticks(())
    idx += 1
axes[1, 0].set_xlabel('Wine Recognition', fontdict={'fontsize': 15})

# ===================================================================================================

n_neighbors = 10

def Malware(seed, batch_size, max_steps): 
    '''
    reward: s_b
    with spectral embedding
    '''
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    malware = pd.read_csv('./datasets/uci_malware_detection.csv')

    X = malware.iloc[:, 1:].to_numpy().astype('float64')
    label = malware.iloc[:, 0].to_numpy()
    enc = OrdinalEncoder()
    label = enc.fit_transform(label[:, np.newaxis]).ravel().astype('int32')
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X, label = shuffle(X, label, random_state=seed)
    
    n, p = X.shape
    K = len(np.unique(label))
    
    embeddings = spectral_embeddings(X, graph, mode, n_neighbors, K)
        
    
    actor = Actor(obs_dim=K, action_dim=K, units=units)
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
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(embeddings)
    for k in range(K):
        dictionary[k] = dict()
        dictionary[k]['mu'] =  np.mean(embeddings[kmeans.labels_ == k, ], axis=0)
        dictionary[k]['sigma'] = np.cov(embeddings[kmeans.labels_ == k, ], rowvar=False)
             
    
    for step in range(max_steps):
#         print('step: ', step)
        
        x_train = get_data(embeddings, batch_size=batch_size)    # batch size小的话，training step就要大一些
        
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
        
        if early_stop & (step > 6):
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
                & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
            
            # if abs(np.mean(r_list[:-10]) - np.mean(r_list[:-5])) < 1e-4:
                print(f'converge at step {step}')
                break
    
    
    with torch.no_grad():
        actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

#     acp_RI, acp_MI, acp_HS, acp_CS  = metrics(label, logits.argmax(dim=1))
    
    results = np.zeros((7, 4))
    
    results[0, :] = metrics(label, logits.argmax(dim=1))
    
    gm = GaussianMixture(n_components=K, n_init=1, random_state=seed).fit(X)
    gm_pred = gm.predict(X)
    results[1, :] = metrics(label, gm_pred)
    
    km = KMeans(n_clusters=K, random_state=seed).fit(X)
    km_pred = km.predict(X)
    results[2, :] = metrics(label, km_pred)
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    sc_pred = sc.labels_
    results[3, :] = metrics(label, sc_pred)
        
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    al_pred = average_linkage.labels_
    results[4, :] = metrics(label, al_pred)
    
    rgcl_pred = RGCL(X, K, max_step=500, seed=seed)
    results[5, :] = metrics(label, rgcl_pred)
    
    
    srgcl_pred = RGCL(X, K, sus_exp=True, max_step=500, seed=seed)
    results[6, :] = metrics(label, srgcl_pred)
    
    # dbscan = cluster.DBSCAN(eps=0.5, min_samples=5).fit(X)
    # y_pred = dbscan.labels_
    # results[5, :] = metrics(label, y_pred)
    
    # optics = cluster.OPTICS(min_samples=5, xi=0.05).fit(X)
    # y_pred = optics.labels_
    # results[6, :] = metrics(label, y_pred)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results, X, label, logits.argmax(dim=1).numpy(), gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred



malware_results, malware, label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred = Malware(seed=0, batch_size=64, max_steps=15000)



acp_pred = np.where(acp_pred==0, -1, acp_pred)
acp_pred = np.where(acp_pred==1, 0, acp_pred)
acp_pred = np.where(acp_pred==-1, 1, acp_pred)

km_pred = np.where(km_pred==0, -1, km_pred)
km_pred = np.where(km_pred==1, 0, km_pred)
km_pred = np.where(km_pred==-1, 1, km_pred)

# al_pred = np.where(al_pred==0, -1, al_pred)
# al_pred = np.where(al_pred==1, 0, al_pred)
# al_pred = np.where(al_pred==-1, 1, al_pred)

rgcl_pred = np.where(rgcl_pred==0, -1, rgcl_pred)
rgcl_pred = np.where(rgcl_pred==1, 0, rgcl_pred)
rgcl_pred = np.where(rgcl_pred==-1, 1, rgcl_pred)

# srgcl_pred = np.where(srgcl_pred==0, -1, srgcl_pred)
# srgcl_pred = np.where(srgcl_pred==1, 0, srgcl_pred)
# srgcl_pred = np.where(srgcl_pred==-1, 1, srgcl_pred)


# par = range(30, 60)
# fig, axes = plt.subplots(5, 6)
# idx = 0
# for i in range(5):
#     for j in range(6):
#         X_embedded = TSNE(n_components=2, perplexity=par[idx], random_state=42).fit_transform(X)    
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)
#         idx += 1


# X_embedded = TSNE(n_components=2, perplexity=64, random_state=42).fit_transform(X) 
malware = pd.read_csv('./datasets/uci_malware_detection.csv')
X = malware.iloc[:, 1:].to_numpy().astype('float64')
label = malware.iloc[:, 0].to_numpy()
enc = OrdinalEncoder()
label = enc.fit_transform(label[:, np.newaxis]).ravel().astype('int32')

scaler = StandardScaler()
X = scaler.fit_transform(X)

malware, label = shuffle(X, label, random_state=0)
X_embedded = KernelPCA(n_components=2, kernel='rbf').fit_transform(malware)
# plt.figure()
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=label)




malware_label_list = [label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred]
# names = ['Ground Truth', 'ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering']

# fig, axes = plt.subplots(2, 3)
# plt.subplots_adjust(top=0.955,
#                     bottom=0.1,
#                     left=0.045,
#                     right=0.985,
#                     hspace=0.35,
#                     wspace=0.2)
# plt.style.use('bmh')
# idx = 0
# for i in range(2):
#     for j in range(3):
#         # plt.style.use('bmh')
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label_list[idx], alpha=1)
#         axes[i, j].set_title(names[idx], fontdict={'fontsize': 20})
#         if i !=0 or j != 0:
#             axes[i, j].set_xlabel(f'Adjusted Rand Index: {results[idx-1, 0]:.2f}, Adjusted Mutual Information: {results[idx-1, 1]:.2f}\nHomogeneity: {results[idx-1, 2]:.2f}, Completeness: {results[idx-1, 3]:.2f}')
#         idx += 1
# axes[0, 0].set_xlabel('Breast cancer wisconsin', fontdict={'fontsize': 15})


idx = 0
for j in range(8):
    axes[2, j].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                        c=malware_label_list[idx], alpha=1, s=14)
    # axes[1, j].set_title(names[idx], fontdict={'fontsize': 15})
    if j != 0:
        axes[2, j].set_xlabel(f'Adjusted RI: {malware_results[idx-1, 0]:.2f}, Adjusted MI: {malware_results[idx-1, 1]:.2f}\nHomogeneity: {malware_results[idx-1, 2]:.2f}, Completeness: {malware_results[idx-1, 3]:.2f}', fontsize=8)
        axes[2, j].set_xticks(())
        axes[2, j].set_yticks(())
    idx += 1
axes[2, 0].set_xlabel('Malware Executable Detection', fontdict={'fontsize': 10})



# ========================================================================

n_neighbors = 20
def Breast(seed, batch_size, max_steps=5000): 
    '''
    reward: s_b
    with spectral embedding
    '''
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    data = datasets.load_breast_cancer()
    X, label = shuffle(data.data, data.target, random_state=seed)
    K = len(np.unique(label))
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    embeddings = spectral_embeddings(X, graph, mode, n_neighbors, K)
        
    
    actor = Actor(obs_dim=K, action_dim=K, units=units)
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
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(embeddings)
    for k in range(K):
        dictionary[k] = dict()
        dictionary[k]['mu'] =  np.mean(embeddings[kmeans.labels_ == k, ], axis=0)
        dictionary[k]['sigma'] = np.cov(embeddings[kmeans.labels_ == k, ], rowvar=False)
             
    
    for step in range(max_steps):
#         print('step: ', step)
        
        x_train = get_data(embeddings, batch_size=batch_size)    # batch size小的话，training step就要大一些
        
        actions, log_probs, _ = actor(x_train)
        
        
        # compute reward
        rewards, _, _, _ = compute_reward(x_train, K, actions.numpy(), log_probs, dictionary, model_based, True)
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
        
        if early_stop & (step > 6):
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
                & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
            
            # if abs(np.mean(r_list[:-10]) - np.mean(r_list[:-5])) < 1e-4:
                print(f'converge at step {step}')
                break
    
    
    with torch.no_grad():
        actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

#     acp_RI, acp_MI, acp_HS, acp_CS  = metrics(label, logits.argmax(dim=1))
    
    results = np.zeros((7, 4))
    
    results[0, :] = metrics(label, logits.argmax(dim=1))
    
    gm = GaussianMixture(n_components=K, n_init=1, random_state=seed).fit(X)
    gm_pred = gm.predict(X)
    results[1, :] = metrics(label, gm_pred)
    
    km = KMeans(n_clusters=K, random_state=seed).fit(X)
    km_pred = km.predict(X)
    results[2, :] = metrics(label, km_pred)
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    sc_pred = sc.labels_
    results[3, :] = metrics(label, sc_pred)
        
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    al_pred = average_linkage.labels_
    results[4, :] = metrics(label, al_pred)
    
    rgcl_pred = RGCL(X, K, seed=seed)
    results[5, :] = metrics(label, rgcl_pred)
    
    
    srgcl_pred = RGCL(X, K, sus_exp=True, seed=seed)
    results[6, :] = metrics(label, srgcl_pred)
    
    # dbscan = cluster.DBSCAN(eps=0.5, min_samples=5).fit(X)
    # y_pred = dbscan.labels_
    # results[5, :] = metrics(label, y_pred)
    
    # optics = cluster.OPTICS(min_samples=5, xi=0.05).fit(X)
    # y_pred = optics.labels_
    # results[6, :] = metrics(label, y_pred)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results, X, label, logits.argmax(dim=1).numpy(), gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred



breast_results, breast, label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred = Breast(seed=5, batch_size=64)



gm_pred = np.where(gm_pred==0, -1, gm_pred)
gm_pred = np.where(gm_pred==1, 0, gm_pred)
gm_pred = np.where(gm_pred==-1, 1, gm_pred)

km_pred = np.where(km_pred==0, -1, km_pred)
km_pred = np.where(km_pred==1, 0, km_pred)
km_pred = np.where(km_pred==-1, 1, km_pred)

al_pred = np.where(al_pred==0, -1, al_pred)
al_pred = np.where(al_pred==1, 0, al_pred)
al_pred = np.where(al_pred==-1, 1, al_pred)

rgcl_pred = np.where(rgcl_pred==0, -1, rgcl_pred)
rgcl_pred = np.where(rgcl_pred==1, 0, rgcl_pred)
rgcl_pred = np.where(rgcl_pred==-1, 1, rgcl_pred)

srgcl_pred = np.where(srgcl_pred==0, -1, srgcl_pred)
srgcl_pred = np.where(srgcl_pred==1, 0, srgcl_pred)
srgcl_pred = np.where(srgcl_pred==-1, 1, srgcl_pred)


# par = range(30, 60)
# fig, axes = plt.subplots(5, 6)
# idx = 0
# for i in range(5):
#     for j in range(6):
#         X_embedded = TSNE(n_components=2, perplexity=par[idx], random_state=42).fit_transform(X)    
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)
#         idx += 1


# X_embedded = TSNE(n_components=2, perplexity=64, random_state=42).fit_transform(X) 
data = datasets.load_breast_cancer()
X, label = shuffle(data.data, data.target, random_state=5)
scaler = StandardScaler()
breast = scaler.fit_transform(X)
X_embedded = KernelPCA(n_components=2, kernel='rbf').fit_transform(breast)
# plt.figure()
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=label)




breast_label_list = [label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred]
# names = ['Ground Truth', 'ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering']

# fig, axes = plt.subplots(2, 3)
# plt.subplots_adjust(top=0.955,
#                     bottom=0.1,
#                     left=0.045,
#                     right=0.985,
#                     hspace=0.35,
#                     wspace=0.2)
# plt.style.use('bmh')
# idx = 0
# for i in range(2):
#     for j in range(3):
#         # plt.style.use('bmh')
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label_list[idx], alpha=1)
#         axes[i, j].set_title(names[idx], fontdict={'fontsize': 20})
#         if i !=0 or j != 0:
#             axes[i, j].set_xlabel(f'Adjusted Rand Index: {results[idx-1, 0]:.2f}, Adjusted Mutual Information: {results[idx-1, 1]:.2f}\nHomogeneity: {results[idx-1, 2]:.2f}, Completeness: {results[idx-1, 3]:.2f}')
#         idx += 1
# axes[0, 0].set_xlabel('Breast cancer wisconsin', fontdict={'fontsize': 15})


idx = 0
for j in range(8):
    axes[3, j].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                        c=breast_label_list[idx], alpha=1, s=14)
    # axes[1, j].set_title(names[idx], fontdict={'fontsize': 15})
    if j != 0:
        axes[3, j].set_xlabel(f'Adjusted RI: {breast_results[idx-1, 0]:.2f}, Adjusted MI: {breast_results[idx-1, 1]:.2f}\nHomogeneity: {breast_results[idx-1, 2]:.2f}, Completeness: {breast_results[idx-1, 3]:.2f}', fontsize=8)
        axes[3, j].set_xticks(())
        axes[3, j].set_yticks(())
    idx += 1
axes[3, 0].set_xlabel('Breast Cancer Wisconsin', fontdict={'fontsize': 12})


# =================================================================================================
n_neighbors = 10

def RNA(seed, batch_size, max_steps): 
    '''
    reward: s_b
    with spectral embedding
    '''
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    X = pd.read_csv('./datasets/RNA-Seq/data.csv')
    label = pd.read_csv('./datasets/RNA-Seq/labels.csv')
    
    X = X.iloc[:, 1:].to_numpy()
    label = label.iloc[:, 1:]
    enc = OrdinalEncoder()
    label = enc.fit_transform(label.to_numpy()).ravel().astype('int32')
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X, label = shuffle(X, label, random_state=seed)
    
    n, p = X.shape
    K = len(np.unique(label))
    
    embeddings = spectral_embeddings(X, graph, mode, n_neighbors, K)
        
    
    actor = Actor(obs_dim=K, action_dim=K, units=units)
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
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(embeddings)
    for k in range(K):
        dictionary[k] = dict()
        dictionary[k]['mu'] =  np.mean(embeddings[kmeans.labels_ == k, ], axis=0)
        dictionary[k]['sigma'] = np.cov(embeddings[kmeans.labels_ == k, ], rowvar=False)
             
    
    for step in range(max_steps):
#         print('step: ', step)
        
        x_train = get_data(embeddings, batch_size=batch_size)    # batch size小的话，training step就要大一些
        
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
        
        if early_stop & (step > 6):
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
                & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3):
            
            # if abs(np.mean(r_list[:-10]) - np.mean(r_list[:-5])) < 1e-4:
                print(f'converge at step {step}')
                break
    
    
    with torch.no_grad():
        actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

#     acp_RI, acp_MI, acp_HS, acp_CS  = metrics(label, logits.argmax(dim=1))
    
    results = np.zeros((7, 4))
    
    results[0, :] = metrics(label, logits.argmax(dim=1))
    
    gm = GaussianMixture(n_components=K, n_init=1, covariance_type='diag', random_state=seed).fit(X)
    gm_pred = gm.predict(X)
    results[1, :] = metrics(label, gm_pred)
    
    km = KMeans(n_clusters=K, random_state=seed).fit(X)
    km_pred = km.predict(X)
    results[2, :] = metrics(label, km_pred)
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    sc_pred = sc.labels_
    results[3, :] = metrics(label, sc_pred)
        
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    al_pred = average_linkage.labels_
    results[4, :] = metrics(label, al_pred)
    
    rgcl_pred = RGCL(X, K, max_step=500, seed=seed)
    results[5, :] = metrics(label, rgcl_pred)
    
    
    srgcl_pred = RGCL(X, K, sus_exp=True, max_step=500, seed=seed)
    results[6, :] = metrics(label, srgcl_pred)
    
    # dbscan = cluster.DBSCAN(eps=0.5, min_samples=5).fit(X)
    # y_pred = dbscan.labels_
    # results[5, :] = metrics(label, y_pred)
    
    # optics = cluster.OPTICS(min_samples=5, xi=0.05).fit(X)
    # y_pred = optics.labels_
    # results[6, :] = metrics(label, y_pred)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results, X, label, logits.argmax(dim=1).numpy(), gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred



rna_results, rna, label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred = RNA(seed=0, batch_size=64, max_steps=25000)



# gm_pred = np.where(gm_pred==0, -1, gm_pred)
# gm_pred = np.where(gm_pred==1, 0, gm_pred)
# gm_pred = np.where(gm_pred==-1, 1, gm_pred)

# km_pred = np.where(km_pred==0, -1, km_pred)
# km_pred = np.where(km_pred==1, 0, km_pred)
# km_pred = np.where(km_pred==-1, 1, km_pred)

# al_pred = np.where(al_pred==0, -1, al_pred)
# al_pred = np.where(al_pred==1, 0, al_pred)
# al_pred = np.where(al_pred==-1, 1, al_pred)

# rgcl_pred = np.where(rgcl_pred==0, -1, rgcl_pred)
# rgcl_pred = np.where(rgcl_pred==1, 0, rgcl_pred)
# rgcl_pred = np.where(rgcl_pred==-1, 1, rgcl_pred)

# srgcl_pred = np.where(srgcl_pred==0, -1, srgcl_pred)
# srgcl_pred = np.where(srgcl_pred==1, 0, srgcl_pred)
# srgcl_pred = np.where(srgcl_pred==-1, 1, srgcl_pred)


# par = range(30, 60)
# fig, axes = plt.subplots(5, 6)
# idx = 0
# for i in range(5):
#     for j in range(6):
#         X_embedded = TSNE(n_components=2, perplexity=par[idx], random_state=42).fit_transform(X)    
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)
#         idx += 1


# X_embedded = TSNE(n_components=2, perplexity=64, random_state=42).fit_transform(X)
X = pd.read_csv('./datasets/RNA-Seq/data.csv')
label = pd.read_csv('./datasets/RNA-Seq/labels.csv')

X = X.iloc[:, 1:].to_numpy()
label = label.iloc[:, 1:]
enc = OrdinalEncoder()
label = enc.fit_transform(label.to_numpy()).ravel().astype('int32')

scaler = StandardScaler()
X = scaler.fit_transform(X)

rna, label = shuffle(X, label, random_state=0) 
X_embedded = KernelPCA(n_components=2, kernel='rbf').fit_transform(rna)
# plt.figure()
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=label)




rna_label_list = [label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred]
# names = ['Ground Truth', 'ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering']

# fig, axes = plt.subplots(2, 3)
# plt.subplots_adjust(top=0.955,
#                     bottom=0.1,
#                     left=0.045,
#                     right=0.985,
#                     hspace=0.35,
#                     wspace=0.2)
# plt.style.use('bmh')
# idx = 0
# for i in range(2):
#     for j in range(3):
#         # plt.style.use('bmh')
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label_list[idx], alpha=1)
#         axes[i, j].set_title(names[idx], fontdict={'fontsize': 20})
#         if i !=0 or j != 0:
#             axes[i, j].set_xlabel(f'Adjusted Rand Index: {results[idx-1, 0]:.2f}, Adjusted Mutual Information: {results[idx-1, 1]:.2f}\nHomogeneity: {results[idx-1, 2]:.2f}, Completeness: {results[idx-1, 3]:.2f}')
#         idx += 1
# axes[0, 0].set_xlabel('Breast cancer wisconsin', fontdict={'fontsize': 15})


idx = 0
for j in range(8):
    axes[4, j].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                        c=rna_label_list[idx], alpha=1, s=14)
    # axes[1, j].set_title(names[idx], fontdict={'fontsize': 15})
    if j != 0:
        axes[4, j].set_xlabel(f'Adjusted RI: {rna_results[idx-1, 0]:.2f}, Adjusted MI: {rna_results[idx-1, 1]:.2f}\nHomogeneity: {rna_results[idx-1, 2]:.2f}, Completeness: {rna_results[idx-1, 3]:.2f}', fontsize=8)
        axes[4, j].set_xticks(())
        axes[4, j].set_yticks(())
    idx += 1
axes[4, 0].set_xlabel('RNA-Seq PANCAN', fontdict={'fontsize': 15})

# ============================================================================================

n_neighbors = 10

def APS(seed, batch_size, max_steps): 
    '''
    reward: s_b
    with spectral embedding
    '''
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    paths = os.listdir('./datasets/APS')

    aps = pd.DataFrame()
    for path in paths:
        aps = pd.concat([aps, pd.read_csv(os.path.join('./datasets/APS', path))], axis=0)
    
    aps_pos = aps.loc[aps['class']==0.9921875, :] 
    
    idx = np.random.randint(0, 74625, 1375)
    aps_neg = aps.iloc[idx, :]
    
    aps = pd.concat([aps_neg, aps_pos], axis=0)
    
    X = aps.iloc[:, 1:].to_numpy()
    label = aps.iloc[:, 0].to_numpy()
    label = np.where(label==-0.9921875, 0, 1)
    
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    
    X, label = shuffle(X, label, random_state=seed)
    
    n, p = X.shape
    K = len(np.unique(label))
    
    # embeddings = spectral_embeddings(X, graph, mode, n_neighbors, K)
        
    
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
        
        x_train = get_data(X, batch_size=batch_size)    # batch size小的话，training step就要大一些
        
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

#     acp_RI, acp_MI, acp_HS, acp_CS  = metrics(label, logits.argmax(dim=1))
    
    results = np.zeros((7, 4))
    
    results[0, :] = metrics(label, logits.argmax(dim=1))
    
    gm = GaussianMixture(n_components=K, n_init=1, random_state=seed).fit(X)
    gm_pred = gm.predict(X)
    results[1, :] = metrics(label, gm_pred)
    
    km = KMeans(n_clusters=K, random_state=seed).fit(X)
    km_pred = km.predict(X)
    results[2, :] = metrics(label, km_pred)
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    sc_pred = sc.labels_
    results[3, :] = metrics(label, sc_pred)
        
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    al_pred = average_linkage.labels_
    results[4, :] = metrics(label, al_pred)
    
    rgcl_pred = RGCL(X, K, max_step=500, seed=seed)
    results[5, :] = metrics(label, rgcl_pred)
    
    
    srgcl_pred = RGCL(X, K, sus_exp=True, max_step=500, seed=seed)
    results[6, :] = metrics(label, srgcl_pred)
    
    # dbscan = cluster.DBSCAN(eps=0.5, min_samples=5).fit(X)
    # y_pred = dbscan.labels_
    # results[5, :] = metrics(label, y_pred)
    
    # optics = cluster.OPTICS(min_samples=5, xi=0.05).fit(X)
    # y_pred = optics.labels_
    # results[6, :] = metrics(label, y_pred)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results, X, label, logits.argmax(dim=1).numpy(), gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred



aps_results, aps, label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred = APS(seed=0, batch_size=128, max_steps=15000)



label = np.where(label==0, -1, label)
label = np.where(label==1, 0, label)
label = np.where(label==-1, 1, label)

# km_pred = np.where(km_pred==0, -1, km_pred)
# km_pred = np.where(km_pred==1, 0, km_pred)
# km_pred = np.where(km_pred==-1, 1, km_pred)

# al_pred = np.where(al_pred==0, -1, al_pred)
# al_pred = np.where(al_pred==1, 0, al_pred)
# al_pred = np.where(al_pred==-1, 1, al_pred)

# rgcl_pred = np.where(rgcl_pred==0, -1, rgcl_pred)
# rgcl_pred = np.where(rgcl_pred==1, 0, rgcl_pred)
# rgcl_pred = np.where(rgcl_pred==-1, 1, rgcl_pred)

# srgcl_pred = np.where(srgcl_pred==0, -1, srgcl_pred)
# srgcl_pred = np.where(srgcl_pred==1, 0, srgcl_pred)
# srgcl_pred = np.where(srgcl_pred==-1, 1, srgcl_pred)


# par = range(30, 60)
# fig, axes = plt.subplots(5, 6)
# idx = 0
# for i in range(5):
#     for j in range(6):
#         X_embedded = TSNE(n_components=2, perplexity=par[idx], random_state=42).fit_transform(X)    
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)
#         idx += 1


# X_embedded = TSNE(n_components=2, perplexity=64, random_state=42).fit_transform(X) 
paths = os.listdir('./datasets/APS')

aps = pd.DataFrame()
for path in paths:
    aps = pd.concat([aps, pd.read_csv(os.path.join('./datasets/APS/', path))], axis=0)

aps_pos = aps.loc[aps['class']==0.9921875, :] 

idx = np.random.randint(0, 74625, 1375)
aps_neg = aps.iloc[idx, :]

aps = pd.concat([aps_neg, aps_pos], axis=0)

X = aps.iloc[:, 1:].to_numpy()
label = aps.iloc[:, 0].to_numpy()
label = np.where(label==-0.9921875, 0, 1)

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

aps, label = shuffle(X, label, random_state=0)
X_embedded = KernelPCA(n_components=2, kernel='rbf').fit_transform(aps)
# plt.figure()
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=label)


# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=label)




aps_label_list = [label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred]
# names = ['Ground Truth', 'ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering']

# fig, axes = plt.subplots(2, 3)
# plt.subplots_adjust(top=0.955,
#                     bottom=0.1,
#                     left=0.045,
#                     right=0.985,
#                     hspace=0.35,
#                     wspace=0.2)
# plt.style.use('bmh')
# idx = 0
# for i in range(2):
#     for j in range(3):
#         # plt.style.use('bmh')
#         axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1], c=label_list[idx], alpha=1)
#         axes[i, j].set_title(names[idx], fontdict={'fontsize': 20})
#         if i !=0 or j != 0:
#             axes[i, j].set_xlabel(f'Adjusted Rand Index: {results[idx-1, 0]:.2f}, Adjusted Mutual Information: {results[idx-1, 1]:.2f}\nHomogeneity: {results[idx-1, 2]:.2f}, Completeness: {results[idx-1, 3]:.2f}')
#         idx += 1
# axes[0, 0].set_xlabel('Breast cancer wisconsin', fontdict={'fontsize': 15})


idx = 0
for j in range(8):
    axes[5, j].scatter(X_embedded[:, 0], X_embedded[:, 1], 
                        c=aps_label_list[idx], alpha=1, s=14)
    # axes[1, j].set_title(names[idx], fontdict={'fontsize': 15})
    if j != 0:
        axes[5, j].set_xlabel(f'Adjusted RI: {aps_results[idx-1, 0]:.2f}, Adjusted MI: {aps_results[idx-1, 1]:.2f}\nHomogeneity: {aps_results[idx-1, 2]:.2f}, Completeness: {aps_results[idx-1, 3]:.2f}', fontsize=8)
        axes[5, j].set_xticks(())
        axes[5, j].set_yticks(())
    idx += 1
axes[5, 0].set_xlabel('APS Failure', fontdict={'fontsize': 15})






# label, acp_pred, gm_pred, km_pred, sc_pred, al_pred, rgcl_pred, srgcl_pred = aps_label_list

# aps_results = np.zeros((7, 4))
# aps_results[0, :] = metrics(label, acp_pred)

# aps_results[1, :] = metrics(label, gm_pred)

# aps_results[2, :] = metrics(label, km_pred)

# aps_results[3, :] = metrics(label, sc_pred)
    
# aps_results[4, :] = metrics(label, al_pred)

# aps_results[5, :] = metrics(label, rgcl_pred)

# aps_results[6, :] = metrics(label, srgcl_pred)







# # ============================================================================================
# dats = []
# dats.append(iris_label_list)
# dats.append(wine_label_list)
# dats.append(malware_label_list)
# dats.append(breast_label_list)
# dats.append(rna_label_list)
# dats.append(aps_label_list)

# dats = np.array([dat for dat in dats])

# np.save('label_list.npy', dats)


# dats = np.load('label_list.npy', allow_pickle=True)
# iris_label_list = dats[0, :]
# wine_label_list = dats[1, :]
# malware_label_list = dats[2, :]
# breast_label_list = dats[3, :]
# rna_label_list = dats[4, :]
# aps_label_list = dats[5, :]


# dats = np.array([iris_results, wine_results, malware_results, breast_results, rna_results, aps_results])
# np.save('result_list.npy', dats)


# dats = np.load('result_list.npy')
# iris_results = dats[0, :, :]
# wine_results = dats[1, :, :] 
# malware_results = dats[2, :, ::] 
# breast_results = dats[3, :, :]
# rna_results = dats[4, :, :] 
# aps_results = dats[5, :, :]

# data_sets = ['Iris Plants', 'Wine Recognition', 'Malware Executable Detection', 'Breast Cancer Wisconsin', 'RNA-Seq PANCAN', 'APS Failure']