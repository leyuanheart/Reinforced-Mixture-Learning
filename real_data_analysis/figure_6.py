import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import os
import sys


paths = os.listdir('./results')

# 0 'breast_rds5-10.npy'
# 1 'iris_rds20-25.npy'
# 2 'wine_rds5-10.npy'


x_ticks = np.array([100, 200, 300, 400])
width = 15
metric_names = ['Adjusted\nRand Index', 'Adjusted\nMutual Information', 
                'Homogeneity', 'Completeness']
method_names = ['ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering']

data_names = ['Iris plants', 'Wine recognition', 'Breast cancer wisconsin']


fig = plt.figure()     # figsize=(10, 10)
plt.style.use('bmh')
plt.subplots_adjust(top=0.928,
                    bottom=0.087,
                    left=0.022,
                    right=0.992,
                    hspace=0.2,
                    wspace=0.071)
i = 0
idx_list = [1, 2, 0]
for i in range(3):
    ax = plt.subplot(1,3,int('{}'.format(i+1)))    
    dats = np.load(os.path.join('./data/real_data', paths[idx_list[i]]))    
    dats = np.array(dats)     # (repetitions, method_types, metrics)        
    means = dats.mean(axis=0)    
    stds = dats.std(axis=0)
    rects1 = ax.bar(x_ticks - 2*width, means[0, :], width, yerr=stds[0, :], label=method_names[0])
    rects2 = ax.bar(x_ticks - width, means[1, :], width, yerr=stds[1, :], label=method_names[1])
    rects3 = ax.bar(x_ticks, means[2, :], width, yerr=stds[2, :], label=method_names[2])
    rects4 = ax.bar(x_ticks + width, means[3, :], width, yerr=stds[3, :], label=method_names[3])
    rects5 = ax.bar(x_ticks + 2*width, means[4, :], width, yerr=stds[4, :], label=method_names[4])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(metric_names) # , fontdict={'fontsize': 20}
    ax.set_ylim(0, 1)
    # ax.set_yticks([0, 0.5, 0.8, 0.9, 1])
    ax.grid(False, axis='x')        
    ax.set_title(data_names[i], fontdict={'fontsize': 20})            
ax.legend(loc='best')