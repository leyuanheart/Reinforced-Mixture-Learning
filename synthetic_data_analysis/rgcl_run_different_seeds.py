# -*- coding: utf-8 -*-
"""
Created on Fri May 13 15:36:32 2022

@author: leyuan
"""

import numpy as np
import time
import datetime

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from rgcl import RGCL

import multiprocessing as mp
from pprint import pprint
from tqdm import tqdm


def metrics(label, y_pred):
    return adjusted_rand_score(label, y_pred), adjusted_mutual_info_score(label, y_pred), homogeneity_score(label, y_pred), completeness_score(label, y_pred)



def run(seed): 
    start = time.time()
    print(f'random seed: {seed} is running')
    np.random.seed(seed)    
    
    X, label = datasets.make_circles(n_samples=n, factor=0.5, noise=0.05, random_state=seed)
    # X, label = datasets.make_moons(n_samples=n, noise=0.05, random_state=seed)
    # X, label = datasets.make_blobs(n_samples=n, centers=3, cluster_std=1, random_state=seed)
    # X, label = datasets.make_blobs(n_samples=n, random_state=seed)
    # transformation = [[0.6, -0.6], [-0.4, 0.8]]
    # X = np.dot(X, transformation)
    # X, label = datasets.make_blobs(n_samples=n, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
    # X, label = make_classification(n_samples=n, n_features=p, n_informative=p, n_redundant=0, n_repeated=0,
    #                                 n_classes=K, n_clusters_per_class=1, class_sep=2, random_state=seed)
    
    # data = datasets.load_iris()                                    # 20-25
    # X, label = shuffle(data.data, data.target, random_state=seed)  
    
    # data = datasets.load_wine()                                    # 5-10
    # data = datasets.load_breast_cancer()                           # 5-10
    # X, label = shuffle(data.data, data.target, random_state=seed)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)                                    
    
    
    y_rgcl = RGCL(X, K, seed=seed)
    y_srgcl = RGCL(X, K, sus_exp=True, seed=seed)
    
    results = np.zeros((2, 4))
    
    results[0, :] = metrics(label, y_rgcl)
    results[1, :] = metrics(label, y_srgcl)
    
    end = time.time()
    print(f'rd: {seed} take {datetime.timedelta(seconds = end - start)}')
    
    return results



n = 200
p = 2
K = 3


if __name__ == '__main__':   
    start = time.time()
    dats = []
    for sd in tqdm(range(5, 10)):
        dats.append(run(sd))

    # print("CPU的核数为：{}".format(mp.cpu_count()))
    
#     pool = mp.Pool(5)
#     dats = pool.map(run, range(50))
#     pool.close()
    end = time.time()
    print(datetime.timedelta(seconds = end - start))
    
    
    dats = np.array([dat for dat in dats])

    
    
pprint(dats.mean(axis=0).round(3))
pprint(dats.std(axis=0).round(3))



np.save('./results/rgcl_results/circle.npy', dats)



















































