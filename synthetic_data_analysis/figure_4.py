import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import os
import sys


# ================== model-free simulation results ================


paths = os.listdir('./results/model_free')


# 0 anisotropic_blobs.npy
# 1 make_blobs.npy
# 2 make_circles.npy
# 3 make_classes.npy
# 4 make_moons.npy
# 5 varied_variances_blobs.npy


x_labels = [300, 600, 900, 1200]
metric_names = ['Adjusted\nRand Index', 'Adjusted\nMutual Information', 
                'Homogeneity', 'Completeness']
method_names = ['ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering', 'RGCL', 'SRGCL']
data_names = ['Circles', 'Moons', 'Blobs', 'Anisotropicly distributed blobs', 'Blobs with varied variances', 'Make classification']

idx_list = [2, 4, 1, 0, 5, 3]


def figure_4():
    fig, axes = plt.subplots(2, 3, sharey=True)  # , figsize=(10, 10)
    plt.style.use('bmh')
    plt.subplots_adjust(top=0.95,
                        bottom=0.05,
                        left=0.11,
                        right=0.9,
                        hspace=0.3,
                        wspace=0.2)

    idx = 0 
    for i in range(2):
        for j in range(3):
            dats = np.load(os.path.join('./results/model_free', paths[idx_list[idx]]))    
            dats = np.array(dats) # (repetitions, method_types, metrics)        
            means = dats.mean(axis=0)    
            stds = dats.std(axis=0)
            
            axes[i, j].errorbar(np.array(x_labels)-90, means[0, :], yerr=stds[0, :], fmt='o', elinewidth=5, ms=9, label=method_names[0])
            axes[i, j].errorbar(np.array(x_labels)-60, means[1, :], yerr=stds[1, :], fmt='o', elinewidth=5, ms=9, label=method_names[1])
            axes[i, j].errorbar(np.array(x_labels)-30, means[2, :], yerr=stds[2, :], fmt='o', elinewidth=5, ms=9, label=method_names[2])
            axes[i, j].errorbar(np.array(x_labels), means[3, :], yerr=stds[3, :], fmt='o', elinewidth=5, ms=9, label=method_names[3])
            axes[i, j].errorbar(np.array(x_labels)+30, means[4, :], yerr=stds[4, :], fmt='o', elinewidth=5, ms=9, label=method_names[4])
            axes[i, j].errorbar(np.array(x_labels)+60, means[5, :], yerr=stds[5, :], fmt='o', elinewidth=5, ms=9, label=method_names[5])
            axes[i, j].errorbar(np.array(x_labels)+90, means[6, :], yerr=stds[6, :], fmt='o', elinewidth=5, ms=9, label=method_names[6], c='red')
            
            axes[i, j].set_xticks(x_labels)
            axes[i, j].set_xticklabels(metric_names)
            axes[i, j].set_xlabel('')
            axes[i, j].set_ylabel('')
            axes[i, j].set_ylim(top=1)
            axes[i, j].set_title(data_names[idx], fontdict={'fontsize': 20})
            
            idx += 1
    axes[0, 0].legend(loc='center right')
    axes[0, 2].legend(loc='lower right')



figure_4()














