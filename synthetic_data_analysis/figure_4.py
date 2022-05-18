# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:01:41 2022

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import random
import copy
import warnings

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification
from sklearn.neighbors import DistanceMetric    # 0.17版的sklearn, 新版的在metrics下
from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances
from sklearn.manifold import spectral_embedding
from sklearn.preprocessing import StandardScaler

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


def observed_loglikelihood(x, pi, mu, sigma):
    log_p = []
    for i in range(x.shape[0]):
        l_i = [p * multivariate_normal.pdf(x[i, :], m, s) for p, m, s in zip(pi, mu, sigma)]
        log_p.append(np.log(np.sum(l_i)))
    return log_p


def compute_reward(x, K, actions, log_probs, dictionary, model_based=True):
    '''
    dictionary的作用是为了防止出现某一类别没有被分配数据，无法计算均值和方差的情况。
    这时赋值dictionary中的数值给相应的类。
    考虑dictionary是不是可以在训练中不断地改善？
    '''
    _, p = x.shape
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
        
        if (actions == k).sum() <= p+1:    # 变量维数
            # print('cov can not be calculated')
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
            dist = DistanceMetric.get_metric('euclidean')  # 'mahalanobis', V=sigma_hat[k]; 'euclidean'
            if (actions == k).sum() == 0:
                s_wk = 0
            else:
                s_wk = np.sum(dist.pairwise(x[actions==k,:], np.array([mu_hat[k]])))  # 这个是马氏距离开根号后的求和
            # [(w - mu_hat[k]).dot(np.linalg.inv(sigma_hat[k])).dot((w - mu_hat[k]).T) for w in x[actions==k, :]]
            s_w += pi_hat[k] * s_wk
        # inter-class distance
        s_b = 0
        for l in range(K):
            for j in range(l, K):
                s_b += np.linalg.norm(mu_hat[l]-mu_hat[j])
        
        for _ in range(actions.size):
            reward_list.append(s_b - s_w)   # 减的效果比除要好
        
#         for i, action in enumerate(actions):
#             dist = np.linalg.norm(x[i, :] - mu_hat[action])
#             reward_list.append(-dist)
            
    return -np.array(reward_list), pi_hat, mu_hat, sigma_hat



def metrics(label, y_pred):
    return adjusted_rand_score(label, y_pred), adjusted_mutual_info_score(label, y_pred)


def spectral_embeddings(X, graph, mode, n_neighbors, k):
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
    L, degree = csgraph_laplacian(similarity_graph, normed=False, return_diag=True)
    
    # ========================== eigen decompostion of the laplacian =================
    if isinstance(L, sparse.coo.coo_matrix):
        L = L.todense()
    e, evecs = eigh(L)
    
    embeddings = evecs[:, :k]
    
    return embeddings



def train(seed, dat, K, units=32, lr=1e-3, batch_size=32, max_steps=5000, model_based=False, early_stop=False):  
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    

    n, p = dat.shape
    start = time.time()
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
    kmeans = KMeans(n_clusters=K, random_state=seed).fit(dat)
    for k in range(K):
        dictionary[k] = dict()
        dictionary[k]['mu'] =  np.mean(dat[kmeans.labels_ == k, ], axis=0)
        dictionary[k]['sigma'] = np.cov(dat[kmeans.labels_ == k, ], rowvar=False)
             
    
    for step in range(max_steps):
        # print('step: ', step)
        
        x_train = get_data(dat, batch_size=batch_size)    # batch size小的话，training step就要大一些
        
        actions, log_probs, _ = actor(x_train)
        
        
        # compute reward
        rewards, _, _, _ = compute_reward(x_train, K, actions.numpy(), log_probs, dictionary, model_based)
        r_list.append(rewards.mean())
        # print(f'average reward: {rewards.mean()}')
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
    
    end = time.time()
    
    print(datetime.timedelta(seconds=end - start))
    
    return actor, r_list, dictionary


n_samples = 200
p = 2

graph = 'k_nearest'  # k_nearest, kernel_based
mode = 'connectivity'    # distance, connectivity
n_neighbors = 10

units = 256
batch_size = 32
lr = 5e-3
max_steps = 5000

noisy_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=3)
noisy_moons = make_moons(n_samples=n_samples, noise=0.1, random_state=6)
blobs = make_blobs(n_samples=n_samples, centers=3, cluster_std=1, random_state=6)

# Anisotropicly distributed data
X, y = make_blobs(n_samples=n_samples, random_state=9)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=1)


mc = make_classification(n_samples=n_samples, n_features=p, n_informative=p, n_redundant=0, n_repeated=0,
                    n_classes=3, n_clusters_per_class=1, class_sep=1.5, random_state=10)


data_sets = [noisy_circles, noisy_moons, blobs, aniso, varied, mc]
data_names = ['Circles', 'Moons', 'Blobs', 'Anisotropicly\ndistributed blobs', 'Blobs with\nvaried variances', 'Make classification']
k_list = [2, 2, 3, 3, 3, 3]
seeds = [3, 6, 6, 9, 1, 10]


def figure_4():
    pass

fig, axes = plt.subplots(6, 8, figsize=(15, 45))
plt.subplots_adjust(
                    top=0.955,
                    bottom=0.03,
                    left=0.07,
                    right=0.925,
                    hspace=0.2,
                    wspace=0.125)

for i in range(6):
    X, label = data_sets[i]    
    axes[i, 0].scatter(X[:,0], X[:, 1], c=label, s=15)
    axes[i, 0].grid(False)
    axes[i, 0].set_ylabel(f'{data_names[i]}') # , fontdict={'size': 10}
    
    embeddings = spectral_embeddings(X, graph, mode, n_neighbors, k_list[i])
    
    learning_rate = 1e-3 if i == 4 else lr
    actor, r_list, dictionary = train(seeds[i], embeddings, K=k_list[i], units=units, lr=learning_rate, batch_size=batch_size, max_steps=max_steps, model_based=False, early_stop=False)
    
    with torch.no_grad():
        actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

    # GMM
    gm = GaussianMixture(n_components=k_list[i], n_init=1, random_state=seeds[i]).fit(X)
    # K-Means
    km = KMeans(n_clusters=k_list[i], random_state=seeds[i]).fit(X)
    # Spectral Clustering
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the "
            + "connectivity matrix is [0-9]{1,2}"
            + " > 1. Completing it to avoid stopping the tree early.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding"
            + " may not work as expected.",
            category=UserWarning,
        )
        sc = cluster.SpectralClustering(n_clusters=k_list[i], n_components=k_list[i], affinity='nearest_neighbors', random_state=seeds[i]).fit(X)  
    # Agglomerative Clustering
    average_linkage = cluster.AgglomerativeClustering(linkage="average",
                                                      affinity="cityblock",
                                                      n_clusters=k_list[i],
                                                      ).fit(X)
    y_rgcl = RGCL(X, k_list[i], seed=seeds[i])
    y_srgcl = RGCL(X, k_list[i], sus_exp=True, seed=seeds[i])
    
    
    axes[i, 1].scatter(X[:,0], X[:, 1], c=logits.argmax(dim=1), s=15)  
    axes[i, 1].set_xticks(())
    axes[i, 1].set_yticks(())
    axes[i, 2].scatter(X[:,0], X[:, 1], c=gm.predict(X), s=15)  
    axes[i, 2].set_xticks(())
    axes[i, 2].set_yticks(())
    axes[i, 3].scatter(X[:,0], X[:, 1], c=km.labels_, s=15)   
    axes[i, 3].set_xticks(())
    axes[i, 3].set_yticks(())
    axes[i, 4].scatter(X[:,0], X[:, 1], c=sc.labels_, s=15)
    axes[i, 4].set_xticks(())
    axes[i, 4].set_yticks(())    
    axes[i, 5].scatter(X[:,0], X[:, 1], c=average_linkage.labels_, s=15)
    axes[i, 5].set_xticks(())
    axes[i, 5].set_yticks(())
    axes[i, 6].scatter(X[:,0], X[:, 1], c=y_rgcl, s=15)
    axes[i, 6].set_xticks(())
    axes[i, 6].set_yticks(())
    axes[i, 7].scatter(X[:,0], X[:, 1], c=y_srgcl, s=15)
    axes[i, 7].set_xticks(())
    axes[i, 7].set_yticks(())
            
axes[0, 0].set_title('Ground Truth')
axes[0, 1].set_title('ACE')
axes[0, 2].set_title('GMM')
axes[0, 3].set_title('K-Means')
axes[0, 4].set_title('Spectral\nClustering')
axes[0, 5].set_title('Agglomerative\nClustering')
axes[0, 6].set_title('RGCL')    
axes[0, 7].set_title('SRGCL')





# fig, axes = plt.subplots(6, 8) # , figsize=(15, 45)
# # plt.subplots_adjust(
# #                     top=0.955,
# #                     bottom=0.03,
# #                     left=0.07,
# #                     right=0.925,
# #                     hspace=0.2,
# #                     wspace=0.125)



# i = 0
# X, label = data_sets[i]    
# axes[i, 0].scatter(X[:,0], X[:, 1], c=label, s=15)
# axes[i, 0].grid(False)
# axes[i, 0].set_ylabel(f'{data_names[i]}') # , fontdict={'size': 10}

# embeddings = spectral_embeddings(X, graph, mode, n_neighbors, k_list[i])

# learning_rate = 1e-3 if i == 4 else lr
# actor, r_list, dictionary = train(seeds[i], embeddings, K=k_list[i], units=units, lr=learning_rate, batch_size=batch_size, max_steps=max_steps, model_based=False, early_stop=False)

# with torch.no_grad():
#     actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

# # GMM
# gm = GaussianMixture(n_components=k_list[i], n_init=1, random_state=seeds[i]).fit(X)
# # K-Means
# km = KMeans(n_clusters=k_list[i], random_state=seeds[i]).fit(X)
# # Spectral Clustering
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message="the number of connected components of the "
#         + "connectivity matrix is [0-9]{1,2}"
#         + " > 1. Completing it to avoid stopping the tree early.",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message="Graph is not fully connected, spectral embedding"
#         + " may not work as expected.",
#         category=UserWarning,
#     )
#     sc = cluster.SpectralClustering(n_clusters=k_list[i], n_components=k_list[i], affinity='nearest_neighbors', random_state=seeds[i]).fit(X)  
# # Agglomerative Clustering
# average_linkage = cluster.AgglomerativeClustering(linkage="average",
#                                                     affinity="cityblock",
#                                                     n_clusters=k_list[i],
#                                                     ).fit(X)
# y_rgcl = RGCL(X, k_list[i], seed=seeds[i])
# y_srgcl = RGCL(X, k_list[i], sus_exp=True, seed=seeds[i])




# axes[i, 1].scatter(X[:,0], X[:, 1], c=logits.argmax(dim=1), s=15)  
# axes[i, 1].set_xticks(())
# axes[i, 1].set_yticks(())
# axes[i, 2].scatter(X[:,0], X[:, 1], c=gm.predict(X), s=15)  
# axes[i, 2].set_xticks(())
# axes[i, 2].set_yticks(())
# axes[i, 3].scatter(X[:,0], X[:, 1], c=km.labels_, s=15)   
# axes[i, 3].set_xticks(())
# axes[i, 3].set_yticks(())

# tmp = sc.labels_
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==1, 0, tmp)
# tmp = np.where(tmp==3, 1, tmp)
# axes[i, 4].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 4].set_xticks(())
# axes[i, 4].set_yticks(())  

# axes[i, 5].scatter(X[:,0], X[:, 1], c=average_linkage.labels_, s=15)
# axes[i, 5].set_xticks(())
# axes[i, 5].set_yticks(())
# axes[i, 6].scatter(X[:,0], X[:, 1], c=y_rgcl, s=15)
# axes[i, 6].set_xticks(())
# axes[i, 6].set_yticks(())


# axes[i, 7].scatter(X[:,0], X[:, 1], c=y_srgcl, s=15)
# axes[i, 7].set_xticks(())
# axes[i, 7].set_yticks(())
        



# i = 1
# X, label = data_sets[i]    
# axes[i, 0].scatter(X[:,0], X[:, 1], c=label, s=15)
# axes[i, 0].grid(False)
# axes[i, 0].set_ylabel(f'{data_names[i]}') # , fontdict={'size': 10}

# embeddings = spectral_embeddings(X, graph, mode, n_neighbors, k_list[i])

# learning_rate = 1e-3 if i == 4 else lr
# actor, r_list, dictionary = train(seeds[i], embeddings, K=k_list[i], units=units, lr=learning_rate, batch_size=batch_size, max_steps=max_steps, model_based=False, early_stop=False)

# with torch.no_grad():
#     actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

# # GMM
# gm = GaussianMixture(n_components=k_list[i], n_init=1, random_state=seeds[i]).fit(X)
# # K-Means
# km = KMeans(n_clusters=k_list[i], random_state=seeds[i]).fit(X)
# # Spectral Clustering
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message="the number of connected components of the "
#         + "connectivity matrix is [0-9]{1,2}"
#         + " > 1. Completing it to avoid stopping the tree early.",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message="Graph is not fully connected, spectral embedding"
#         + " may not work as expected.",
#         category=UserWarning,
#     )
#     sc = cluster.SpectralClustering(n_clusters=k_list[i], n_components=k_list[i], affinity='nearest_neighbors', random_state=seeds[i]).fit(X)  
# # Agglomerative Clustering
# average_linkage = cluster.AgglomerativeClustering(linkage="average",
#                                                     affinity="cityblock",
#                                                     n_clusters=k_list[i],
#                                                     ).fit(X)
# y_rgcl = RGCL(X, k_list[i], seed=seeds[i])
# y_srgcl = RGCL(X, k_list[i], sus_exp=True, seed=seeds[i])

# axes[i, 1].scatter(X[:,0], X[:, 1], c=logits.argmax(dim=1), s=15)  
# axes[i, 1].set_xticks(())
# axes[i, 1].set_yticks(())
# axes[i, 2].scatter(X[:,0], X[:, 1], c=gm.predict(X), s=15)  
# axes[i, 2].set_xticks(())
# axes[i, 2].set_yticks(())
# axes[i, 3].scatter(X[:,0], X[:, 1], c=km.labels_, s=15)   
# axes[i, 3].set_xticks(())
# axes[i, 3].set_yticks(())
# axes[i, 4].scatter(X[:,0], X[:, 1], c=(sc.labels_-1)*(-1), s=15)
# axes[i, 4].set_xticks(())
# axes[i, 4].set_yticks(())    
# axes[i, 5].scatter(X[:,0], X[:, 1], c=average_linkage.labels_, s=15)
# axes[i, 5].set_xticks(())
# axes[i, 5].set_yticks(())

# tmp = y_rgcl
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==1, 0, tmp)
# tmp = np.where(tmp==3, 1, tmp)
# axes[i, 6].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 6].set_xticks(())
# axes[i, 6].set_yticks(())

# tmp = y_srgcl
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==1, 0, tmp)
# tmp = np.where(tmp==3, 1, tmp)
# axes[i, 7].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 7].set_xticks(())
# axes[i, 7].set_yticks(())



# i = 2
# X, label = data_sets[i]    
# axes[i, 0].scatter(X[:,0], X[:, 1], c=label, s=15)
# axes[i, 0].grid(False)
# axes[i, 0].set_ylabel(f'{data_names[i]}') # , fontdict={'size': 10}

# embeddings = spectral_embeddings(X, graph, mode, n_neighbors, k_list[i])

# learning_rate = 1e-3 if i == 4 else lr
# actor, r_list, dictionary = train(seeds[i], embeddings, K=k_list[i], units=units, lr=learning_rate, batch_size=batch_size, max_steps=max_steps, model_based=False, early_stop=False)

# with torch.no_grad():
#     actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

# # GMM
# gm = GaussianMixture(n_components=k_list[i], n_init=1, random_state=seeds[i]).fit(X)
# # K-Means
# km = KMeans(n_clusters=k_list[i], random_state=seeds[i]).fit(X)
# # Spectral Clustering
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message="the number of connected components of the "
#         + "connectivity matrix is [0-9]{1,2}"
#         + " > 1. Completing it to avoid stopping the tree early.",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message="Graph is not fully connected, spectral embedding"
#         + " may not work as expected.",
#         category=UserWarning,
#     )
#     sc = cluster.SpectralClustering(n_clusters=k_list[i], n_components=k_list[i], affinity='nearest_neighbors', random_state=seeds[i]).fit(X)  
# # Agglomerative Clustering
# average_linkage = cluster.AgglomerativeClustering(linkage="average",
#                                                     affinity="cityblock",
#                                                     n_clusters=k_list[i],
#                                                     ).fit(X)
# y_rgcl = RGCL(X, k_list[i], seed=42)
# y_srgcl = RGCL(X, k_list[i], sus_exp=True, seed=seeds[i])


# tmp = logits.argmax(dim=1).numpy()
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 1].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 1].set_xticks(())
# axes[i, 1].set_yticks(())

# tmp = gm.predict(X)
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==2, 0, tmp)
# tmp = np.where(tmp==3, 2, tmp)
# axes[i, 2].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 2].set_xticks(())
# axes[i, 2].set_yticks(())


# tmp = km.labels_
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==2, 0, tmp)
# tmp = np.where(tmp==3, 2, tmp)
# axes[i, 3].scatter(X[:,0], X[:, 1], c=tmp, s=15)   
# axes[i, 3].set_xticks(())
# axes[i, 3].set_yticks(())


# tmp = sc.labels_
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==1, 0, tmp)
# tmp = np.where(tmp==3, 1, tmp)
# axes[i, 4].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 4].set_xticks(())
# axes[i, 4].set_yticks(())   


# tmp = average_linkage.labels_
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 5].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 5].set_xticks(())
# axes[i, 5].set_yticks(())


# axes[i, 6].scatter(X[:,0], X[:, 1], c=y_rgcl, s=15)
# axes[i, 6].set_xticks(())
# axes[i, 6].set_yticks(())

# tmp = y_srgcl
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 7].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 7].set_xticks(())
# axes[i, 7].set_yticks(())



# i = 3
# X, label = data_sets[i]    
# axes[i, 0].scatter(X[:,0], X[:, 1], c=label, s=15)
# axes[i, 0].grid(False)
# axes[i, 0].set_ylabel(f'{data_names[i]}') # , fontdict={'size': 10}

# embeddings = spectral_embeddings(X, graph, mode, n_neighbors, k_list[i])

# learning_rate = 1e-3 if i == 4 else lr
# actor, r_list, dictionary = train(seeds[i], embeddings, K=k_list[i], units=units, lr=learning_rate, batch_size=batch_size, max_steps=max_steps, model_based=False, early_stop=False)

# with torch.no_grad():
#     actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

# # GMM
# gm = GaussianMixture(n_components=k_list[i], n_init=1, random_state=seeds[i]).fit(X)
# # K-Means
# km = KMeans(n_clusters=k_list[i], random_state=seeds[i]).fit(X)
# # Spectral Clustering
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message="the number of connected components of the "
#         + "connectivity matrix is [0-9]{1,2}"
#         + " > 1. Completing it to avoid stopping the tree early.",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message="Graph is not fully connected, spectral embedding"
#         + " may not work as expected.",
#         category=UserWarning,
#     )
#     sc = cluster.SpectralClustering(n_clusters=k_list[i], n_components=k_list[i], affinity='nearest_neighbors', random_state=seeds[i]).fit(X)  
# # Agglomerative Clustering
# average_linkage = cluster.AgglomerativeClustering(linkage="average",
#                                                     affinity="cityblock",
#                                                     n_clusters=k_list[i],
#                                                     ).fit(X)
# y_rgcl = RGCL(X, k_list[i], seed=seeds[i])
# y_srgcl = RGCL(X, k_list[i], sus_exp=True, seed=seeds[i])


# tmp = logits.argmax(dim=1).numpy()
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 1].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 1].set_xticks(())
# axes[i, 1].set_yticks(())

# tmp = gm.predict(X)
# axes[i, 2].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 2].set_xticks(())
# axes[i, 2].set_yticks(())


# tmp = km.labels_
# axes[i, 3].scatter(X[:,0], X[:, 1], c=tmp, s=15)   
# axes[i, 3].set_xticks(())
# axes[i, 3].set_yticks(())


# tmp = sc.labels_
# axes[i, 4].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 4].set_xticks(())
# axes[i, 4].set_yticks(())   

 
# tmp = average_linkage.labels_
# tmp -= 1
# tmp = np.where(tmp==-1, 2, tmp)
# axes[i, 5].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 5].set_xticks(())
# axes[i, 5].set_yticks(())


# tmp = y_rgcl
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 6].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 6].set_xticks(())
# axes[i, 6].set_yticks(())


# tmp = y_srgcl
# tmp = np.where(tmp==1, 3, tmp)
# tmp = np.where(tmp==0, 1, tmp)
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 7].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 7].set_xticks(())
# axes[i, 7].set_yticks(())




# i = 4
# X, label = data_sets[i]    
# axes[i, 0].scatter(X[:,0], X[:, 1], c=label, s=15)
# axes[i, 0].grid(False)
# axes[i, 0].set_ylabel(f'{data_names[i]}') # , fontdict={'size': 10}

# embeddings = spectral_embeddings(X, graph, mode, n_neighbors, k_list[i])

# learning_rate = 1e-3 if i == 4 else lr
# actor, r_list, dictionary = train(seeds[i], embeddings, K=k_list[i], units=units, lr=learning_rate, batch_size=batch_size, max_steps=max_steps, model_based=False, early_stop=False)

# with torch.no_grad():
#     actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

# # GMM
# gm = GaussianMixture(n_components=k_list[i], n_init=1, random_state=seeds[i]).fit(X)
# # K-Means
# km = KMeans(n_clusters=k_list[i], random_state=seeds[i]).fit(X)
# # Spectral Clustering
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message="the number of connected components of the "
#         + "connectivity matrix is [0-9]{1,2}"
#         + " > 1. Completing it to avoid stopping the tree early.",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message="Graph is not fully connected, spectral embedding"
#         + " may not work as expected.",
#         category=UserWarning,
#     )
#     sc = cluster.SpectralClustering(n_clusters=k_list[i], n_components=k_list[i], affinity='nearest_neighbors', random_state=seeds[i]).fit(X)  
# # Agglomerative Clustering
# average_linkage = cluster.AgglomerativeClustering(linkage="average",
#                                                     affinity="cityblock",
#                                                     n_clusters=k_list[i],
#                                                     ).fit(X)
# y_rgcl = RGCL(X, k_list[i], seed=seeds[i])
# y_srgcl = RGCL(X, k_list[i], sus_exp=True, seed=seeds[i])


# tmp = logits.argmax(dim=1).numpy()
# axes[i, 1].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 1].set_xticks(())
# axes[i, 1].set_yticks(())

# tmp = gm.predict(X)
# tmp -= 1
# tmp = np.where(tmp==-1, 2, tmp)
# axes[i, 2].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 2].set_xticks(())
# axes[i, 2].set_yticks(())


# tmp = km.labels_
# tmp -= 1
# tmp = np.where(tmp==-1, 2, tmp)
# axes[i, 3].scatter(X[:,0], X[:, 1], c=tmp, s=15)   
# axes[i, 3].set_xticks(())
# axes[i, 3].set_yticks(())


# tmp = sc.labels_
# tmp -= 1
# tmp = np.where(tmp==-1, 2, tmp)
# axes[i, 4].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 4].set_xticks(())
# axes[i, 4].set_yticks(())   


# tmp = average_linkage.labels_
# tmp -= 1
# tmp = np.where(tmp==-1, 2, tmp)
# axes[i, 5].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 5].set_xticks(())
# axes[i, 5].set_yticks(())


# axes[i, 6].scatter(X[:,0], X[:, 1], c=y_rgcl, s=15)
# axes[i, 6].set_xticks(())
# axes[i, 6].set_yticks(())

# axes[i, 7].scatter(X[:,0], X[:, 1], c=y_srgcl, s=15)
# axes[i, 7].set_xticks(())
# axes[i, 7].set_yticks(())




# i = 5
# X, label = data_sets[i]    
# axes[i, 0].scatter(X[:,0], X[:, 1], c=label, s=15)
# axes[i, 0].grid(False)
# axes[i, 0].set_ylabel(f'{data_names[i]}') # , fontdict={'size': 10}

# embeddings = spectral_embeddings(X, graph, mode, n_neighbors, k_list[i])

# learning_rate = 1e-3 if i == 4 else lr
# actor, r_list, dictionary = train(seeds[i], embeddings, K=k_list[i], units=units, lr=learning_rate, batch_size=batch_size, max_steps=max_steps, model_based=False, early_stop=False)

# with torch.no_grad():
#     actions, log_probs, logits = actor(np.ascontiguousarray(embeddings))

# # GMM
# gm = GaussianMixture(n_components=k_list[i], n_init=1, random_state=seeds[i]).fit(X)
# # K-Means
# km = KMeans(n_clusters=k_list[i], random_state=seeds[i]).fit(X)
# # Spectral Clustering
# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         "ignore",
#         message="the number of connected components of the "
#         + "connectivity matrix is [0-9]{1,2}"
#         + " > 1. Completing it to avoid stopping the tree early.",
#         category=UserWarning,
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message="Graph is not fully connected, spectral embedding"
#         + " may not work as expected.",
#         category=UserWarning,
#     )
#     sc = cluster.SpectralClustering(n_clusters=k_list[i], n_components=k_list[i], affinity='nearest_neighbors', random_state=seeds[i]).fit(X)  
# # Agglomerative Clustering
# average_linkage = cluster.AgglomerativeClustering(linkage="average",
#                                                     affinity="cityblock",
#                                                     n_clusters=k_list[i],
#                                                     ).fit(X)
# y_rgcl = RGCL(X, k_list[i], seed=seeds[i])
# y_srgcl = RGCL(X, k_list[i], sus_exp=True, seed=seeds[i])


# tmp = logits.argmax(dim=1).numpy()
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 1].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 1].set_xticks(())
# axes[i, 1].set_yticks(())

# tmp = gm.predict(X)
# tmp = np.where(tmp==1, 3, tmp)
# tmp = np.where(tmp==2, 1, tmp)
# tmp = np.where(tmp==3, 2, tmp)
# axes[i, 2].scatter(X[:,0], X[:, 1], c=tmp, s=15)  
# axes[i, 2].set_xticks(())
# axes[i, 2].set_yticks(())


# tmp = km.labels_
# tmp = np.where(tmp==1, 3, tmp)
# tmp = np.where(tmp==2, 1, tmp)
# tmp = np.where(tmp==3, 2, tmp)
# axes[i, 3].scatter(X[:,0], X[:, 1], c=tmp, s=15)   
# axes[i, 3].set_xticks(())
# axes[i, 3].set_yticks(())


# tmp = sc.labels_
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==2, 0, tmp)
# tmp = np.where(tmp==3, 2, tmp)
# axes[i, 4].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 4].set_xticks(())
# axes[i, 4].set_yticks(())   


# tmp = average_linkage.labels_
# tmp = np.where(tmp==0, 3, tmp)
# tmp = np.where(tmp==2, 0, tmp)
# tmp = np.where(tmp==3, 2, tmp)
# axes[i, 5].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 5].set_xticks(())
# axes[i, 5].set_yticks(())


# tmp = y_rgcl
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 6].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 6].set_xticks(())
# axes[i, 6].set_yticks(())

# tmp = y_srgcl
# tmp += 1
# tmp = np.where(tmp==3, 0, tmp)
# axes[i, 7].scatter(X[:,0], X[:, 1], c=tmp, s=15)
# axes[i, 7].set_xticks(())
# axes[i, 7].set_yticks(())


# axes[0, 0].set_title('Ground Truth')
# axes[0, 1].set_title('ACE')
# axes[0, 2].set_title('GMM')
# axes[0, 3].set_title('K-Means')
# axes[0, 4].set_title('Spectral Clustering')
# axes[0, 5].set_title('Agglomerative Clustering')
# axes[0, 6].set_title('RGCL')
# axes[0, 7].set_title('SRGCL')



