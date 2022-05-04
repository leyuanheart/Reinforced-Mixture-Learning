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

import multiprocessing as mp
from pprint import pprint
from tqdm import tqdm



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



def compute_reward(x, K, actions, log_probs, dictionary, model_based=True):
    '''
    dictionary的作用是为了防止出现某一类别没有被分配数据，无法计算均值和方差的情况。
    这时赋值dictionary中的数值给相应的类。
    考虑dictionary是不是可以在训练中不断地改善？
    '''
    _, dim_x = x.shape
    reward_list = []
    mu_hat = []
    sigma_hat = []
    for k in range(K):
        # print((actions == k).sum())
        if (actions == k).sum() == 0:
            print('mu can not be calculated')
            mu_hat.append(dictionary[k]['mu'])
        else:
            mu_hat.append(np.mean(x[actions == k, :], axis=0))
        
        if (actions == k).sum() <= dim_x+1:    # 变量维数
            print('cov can not be calculated')
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
        
#         for i, action in enumerate(actions):
#             dist = np.linalg.norm(x[i, :] - mu_hat[action])
#             reward_list.append(-dist)
            
    return -np.array(reward_list), pi_hat, mu_hat, sigma_hat


def metrics(label, y_pred):
    return adjusted_rand_score(label, y_pred), adjusted_mutual_info_score(label, y_pred), homogeneity_score(label, y_pred), completeness_score(label, y_pred)

def observed_loglikelihood(x, pi, mu, sigma):
    log_p = []
    for i in range(x.shape[0]):
        l_i = [p * multivariate_normal.pdf(x[i, :], m, s) for p, m, s in zip(pi, mu, sigma)]
        log_p.append(np.log(np.sum(l_i)))
    return log_p


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




n = 200
p = 2
model_based = False
early_stop = False
units = 256
batch_size = 32
lr = 5e-3
max_steps = 5000

graph = 'k_nearest'  # k_nearest, kernel_based
mode = 'connectivity'    # distance, connectivity
n_neighbors = 10


def run(seed): 
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    
    X, label = datasets.make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=seed)
    # X, label = datasets.make_moons(n_samples=n, noise=0.05, random_state=seed)
    # X, label = datasets.make_blobs(n_samples=n, centers=3, cluster_std=1, random_state=seed)
    # X, label = datasets.make_blobs(n_samples=n, random_state=seed)
    # transformation = [[0.6, -0.6], [-0.4, 0.8]]
    # X = np.dot(X, transformation)
    # X, label = datasets.make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    # X, label = make_classification(n_samples=n, n_features=p, n_informative=p, n_redundant=0, n_repeated=0,
    #                                n_classes=K, n_clusters_per_class=1, class_sep=2, random_state=seed)
    

    K = len(np.unique(label))
    embedding_dim = K
    
    
    embeddings = spectral_embeddings(X, graph, mode, n_neighbors, embedding_dim)
    
    actor = Actor(obs_dim=embedding_dim, action_dim=K, units=units)
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
    y_pred = gm.predict(X)
    results[1, :] = metrics(label, y_pred)
    
    km = KMeans(n_clusters=K, random_state=seed).fit(X)
    y_pred = km.predict(X)
    results[2, :] = metrics(label, y_pred)
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    y_pred = sc.labels_
    results[3, :] = metrics(label, y_pred)
    
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    y_pred = average_linkage.labels_
    results[4, :] = metrics(label, y_pred)
    
    dbscan = cluster.DBSCAN(eps=0.5, min_samples=5).fit(X)
    y_pred = dbscan.labels_
    results[5, :] = metrics(label, y_pred)
    
    optics = cluster.OPTICS(min_samples=5, xi=0.05).fit(X)
    y_pred = optics.labels_
    results[6, :] = metrics(label, y_pred)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results



if __name__ == '__main__':   
    start = time.time()
    dats = []
    for sd in tqdm(range(50)):
        dats.append(run(sd))

    # print("CPU的核数为：{}".format(mp.cpu_count()))
    
#     pool = mp.Pool(5)
#     dats = pool.map(run, range(50))
#     pool.close()
    end = time.time()
    print(datetime.timedelta(seconds = end - start))
    
    
    dats = np.array([dat for dat in dats])

    np.save('./results/model_free/n200_p2_K2_make_circles_factor0.5_units256_bz32_lr1e-3_maxiter5000.npy', dats)
    
    pprint(dats.mean(axis=0).round(3))
    pprint(dats.std(axis=0).round(3))