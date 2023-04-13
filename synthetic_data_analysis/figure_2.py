import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import os
import sys



# ================== GMM-setting results ================

x_labels = [100, 200, 300, 400]
metric_names = ['Adjusted Rand Index', 'Adjusted Mutual Information', r'RMSE: $\alpha$', r'RMSE: $\mu$', r'RMSE: $\Sigma$']
method_names = ['ACE', 'EM']
dats = []

dats.append(np.load('./results/model-based/p2_K3/n100.npy'))
dats.append(np.load('./results/model-based/p2_K3/n200.npy'))
dats.append(np.load('./results/model-based/p2_K3/n300.npy'))
dats.append(np.load('./results/model-based/p2_K3/n400.npy'))
    

dats = np.array(dats)

means = dats.mean(axis=1)   # (n_samples, method_types, metrics) 
stds = dats.std(axis=1)


def figure_2():
    
    plt.figure(figsize=(15, 10))
    plt.style.use('bmh')
    # grids = plt.GridSpec(nrows=2, ncols=3)     

    ax0 = plt.subplot(2, 2, 1)
    # ax0 = fig.add_subplot(grids[0, 0]) 
    ax0.errorbar(np.array(x_labels)-10, means[:, 0, 0], yerr=stds[:, 0, 0], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax0.errorbar(np.array(x_labels)+10, means[:, 1, 0], yerr=stds[:, 1, 0], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax0.set_xticks(x_labels)
    ax0.set_xlabel('sample size')
    ax0.set_ylabel(metric_names[0])
    ax0.set_ylim(top=1)
    ax0.set_yticks([0.8, 0.85, 0.9, 0.95, 1])
    ax0.legend(loc='best')

    ax1 = plt.subplot(2, 2, 2)
    # ax1 = fig.add_subplot(grids[0, 1]) 
    ax1.errorbar(np.array(x_labels)-10, means[:, 0, 1], yerr=stds[:, 0, 1], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax1.errorbar(np.array(x_labels)+10, means[:, 1, 1], yerr=stds[:, 1, 1], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax1.set_xticks(x_labels)
    ax1.legend()
    ax1.set_xlabel('sample size')
    ax1.set_ylabel(metric_names[1])
    ax1.set_ylim(top=1)
    ax1.set_yticks([0.8, 0.85, 0.9, 0.95, 1])
    ax1.legend(loc='best')

    ax2 = plt.subplot(2, 3, 4)
    # ax2 = fig.add_subplot(grids[1, 0]) 
    ax2.errorbar(np.array(x_labels)-10, means[:, 0, 2], yerr=stds[:, 0, 2], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax2.errorbar(np.array(x_labels)+10, means[:, 1, 2], yerr=stds[:, 1, 2], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax2.set_xticks(x_labels)
    ax2.set_xlabel('sample size')
    ax2.set_ylabel(metric_names[2])
    ax2.set_ylim(bottom=0)
    # ax2.set_yticks([0, 0.04, 0.08, 0.1])
    ax2.legend(loc='best')

    ax3 = plt.subplot(2, 3, 5)
    # ax3 = fig.add_subplot(grids[1, 1]) 
    ax3.errorbar(np.array(x_labels)-10, means[:, 0, 3], yerr=stds[:, 0, 3], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax3.errorbar(np.array(x_labels)+10, means[:, 1, 3], yerr=stds[:, 1, 3], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax3.set_xticks(x_labels)
    ax3.set_xlabel('sample size')
    ax3.set_ylabel(metric_names[3])
    ax3.set_ylim(bottom=0)
    # ax3.set_yticks([0, 5, 10, 15, 20])
    ax3.legend(loc='best')

    ax4 = plt.subplot(2, 3, 6)
    # ax4 = fig.add_subplot(grids[1, 2]) 
    ax4.errorbar(np.array(x_labels)-10, means[:, 0, 4], yerr=stds[:, 0, 4], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax4.errorbar(np.array(x_labels)+10, means[:, 1, 4], yerr=stds[:, 1, 4], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax4.set_xticks(x_labels)
    ax4.set_xlabel('sample size')
    ax4.set_ylabel(metric_names[4])
    # ax4.set_ylim(bottom=0)
    # ax4.set_yticks([10, 15, 20, 25, 30])
    ax4.legend(loc='best')


figure_2()

## For tables 
# i = 0  
# results = pd.DataFrame(np.zeros((2, 5)), index=method_names)  # n=100

# for j in range(5):
#     results.loc['ACE', j] = f'{means[i, 0, j].mean():.3f}'+'$\pm$'+f'{stds[i, 0, j]:.3f}'
#     results.loc["EM", j] = f'{means[i, 1, j].mean():.3f}'+'$\pm$'+f'{stds[i, 1, j]:.3f}'
    
# results.columns = metric_names