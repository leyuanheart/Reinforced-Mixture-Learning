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
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

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



def compute_reward(x, K, actions, log_probs, dictionary, model_based=True):
    
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
           
    if model_based:
    ## model-based reward    
        reward_list = observed_loglikelihood(x, pi_hat, mu_hat, sigma_hat)
    else:
    ## model-free reward
        # # intra-class distance
        # s_w = 0
        # for k in range(K):
        #     dist = DistanceMetric.get_metric('euclidean')
        #     if (actions == k).sum() == 0:
        #         s_wk = 0
        #     else:
        #         s_wk = np.sum(dist.pairwise(x[actions==k,:], np.array([mu_hat[k]])))  
        #     # [(w - mu_hat[k]).dot(np.linalg.inv(sigma_hat[k])).dot((w - mu_hat[k]).T) for w in x[actions==k, :]]
        #     s_w += pi_hat[k] * s_wk
        # # inter-class distance
        # s_b = 0
        # for l in range(K):
        #     for j in range(l, K):
        #         s_b += np.linalg.norm(mu_hat[l]-mu_hat[j])
        
        # for _ in range(actions.size):
        #     reward_list.append(s_b - s_w)   
        
        '''
        In this case, we find just using a simple distance of each sample and its assigned cluster center as the reward can achieve better performance,
        which indicate that more effort should be made to find a proper reward in the future work. 
        '''
        for i, action in enumerate(actions):
            dist = np.linalg.norm(x[i, :] - mu_hat[action])
            reward_list.append(-dist)                           
            
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



data = datasets.load_breast_cancer()

n, p = data.data.shape
K = 2
embedding_dim = K

model_based = False
early_stop = True
units = 256
batch_size = 100
lr = 5e-4
max_steps = 10000

graph = 'k_nearest'  # k_nearest, kernel_based
mode = 'connectivity'    # distance, connectivity
n_neighbors = 15


def run(seed): 
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    
    X, label = shuffle(data.data, data.target, random_state=seed)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
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
        
        if early_stop & (step > 10):
            if (abs(r_list[-1] - r_list[-2]) < 1e-3) & (abs(r_list[-2] - r_list[-3]) < 1e-3) \
                & (abs(r_list[-3] - r_list[-4]) < 1e-3) & (abs(r_list[-4] - r_list[-5]) < 1e-3) \
                & (abs(r_list[-5] - r_list[-6]) < 1e-3) & (abs(r_list[-6] - r_list[-7]) < 1e-3) \
                & (abs(r_list[-7] - r_list[-8]) < 1e-3) & (abs(r_list[-8] - r_list[-9]) < 1e-3):
            
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
    
    sc = cluster.SpectralClustering(n_clusters=K, n_components=K, affinity='nearest_neighbors', n_neighbors=n_neighbors, random_state=seed).fit(X)  # rbf 依然解决不了circle问题
    y_pred = sc.labels_
    results[3, :] = metrics(label, y_pred)
    
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average",
        affinity="cityblock",
        n_clusters=K,
        ).fit(X)
    y_pred = average_linkage.labels_
    results[4, :] = metrics(label, y_pred)
    
    y_rgcl = RGCL(X, K, seed=seed)
    results[5, :] = metrics(label, y_rgcl)
    
    y_srgcl = RGCL(X, K, sus_exp=True, seed=seed)
    results[6, :] = metrics(label, y_srgcl)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results



if __name__ == '__main__':   
    start = time.time()
    # dats = []
    # for sd in tqdm(range(5, 10)):
    #     dats.append(run(sd))

#     print("CPU的核数为：{}".format(mp.cpu_count()))    
    pool = mp.Pool(5)
    dats = pool.map(run, range(5, 10))
    pool.close()
    end = time.time()
    print(datetime.timedelta(seconds = end - start))
    
    
    dats = np.array([dat for dat in dats])


    np.save('./results/breast_cancer.npy', dats)    
        
    pprint(dats.mean(axis=0).round(3))
    pprint(dats.std(axis=0).round(3))


















