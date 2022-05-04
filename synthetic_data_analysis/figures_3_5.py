import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
import os
import sys



# ================== GMM-setting results ================
paths = os.listdir('./results/GMM')

# 0 'n100_p10_K3_sigma2.5_units256_bz100_lr1e-3_maxiter2500.npy'
# 1 'n100_p2_K3_units256_bz100_lr1e-3_maxiter5000.npy'
# 2 'n100_p3_K4_sigma2_units256_bz100_lr1e-3_maxiter5000.npy'
# 3 'n200_p10_K3_sigma2.5_units256_bz100_lr1e-3_maxiter2500.npy'
# 4 'n200_p2_K3_units256_bz100_lr1e-3_maxiter5000.npy'
# 5 'n200_p3_K4_sigma2_units256_bz100_lr1e-3_maxiter5000.npy'
# 6 'n300_p10_K3_sigma2.5_units256_bz100_lr1e-3_maxiter2500.npy'
# 7 'n300_p2_K3_units256_bz100_lr1e-3_maxiter5000.npy'
# 8 'n300_p3_K4_sigma2_units256_bz100_lr1e-3_maxiter5000.npy'
# 9 'n400_p10_K3_sigma2.5_units256_bz100_lr1e-3_maxiter2500.npy'
# 10 'n400_p2_K3_units256_bz100_lr1e-3_maxiter5000.npy'
# 11 'n400_p3_K4_sigma2_units256_bz100_lr1e-3_maxiter5000.npy'


# p2_K_3_units256_bz100_lr1e-3_maxiter5000            1, 4, 7, 10
# p3_K_4_sigma2_units256_bz100_lr1e-3_maxiter5000     2, 5, 8, 11
# p10_K_3_sigma2.5_units256_bz100_lr1e-3_maxiter2500  0, 3, 6, 9

def figure_3():
    idx = [1, 4, 7, 10]
    # for i in idx:
    #     print(paths[i])
    dats = []
    for i in idx:
        dats.append(np.load(os.path.join('./results/GMM', paths[i])))
        
    dats = np.array(dats)


    x_labels = [100, 200, 300, 400]
    metric_names = ['Adjusted Rand Index', 'Adjusted Mutual Information', r'RMSE: $\alpha$', r'RMSE: $\mu$', r'RMSE: $\Sigma$']
    method_names = ['ACE', 'EM']


    means = dats.mean(axis=1)   # (n_samples, method_types, metrics) 
    stds = dats.std(axis=1)


    plt.style.use('bmh')
    fig = plt.figure(figsize=(15, 10))
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
    # ax1.set_yticks([0.8, 0.85, 0.9, 0.95, 1])
    ax1.legend(loc='best')

    ax2 = plt.subplot(2, 3, 4)
    # ax2 = fig.add_subplot(grids[1, 0]) 
    ax2.errorbar(np.array(x_labels)-10, means[:, 0, 2], yerr=stds[:, 0, 2], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax2.errorbar(np.array(x_labels)+10, means[:, 1, 2], yerr=stds[:, 1, 2], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax2.set_xticks(x_labels)
    ax2.set_xlabel('sample size')
    ax2.set_ylabel(metric_names[2])
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='best')

    ax3 = plt.subplot(2, 3, 5)
    # ax3 = fig.add_subplot(grids[1, 1]) 
    ax3.errorbar(np.array(x_labels)-10, means[:, 0, 3], yerr=stds[:, 0, 3], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax3.errorbar(np.array(x_labels)+10, means[:, 1, 3], yerr=stds[:, 1, 3], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax3.set_xticks(x_labels)
    ax3.set_xlabel('sample size')
    ax3.set_ylabel(metric_names[3])
    ax3.set_ylim(bottom=0)
    ax3.legend(loc='best')

    ax4 = plt.subplot(2, 3, 6)
    # ax4 = fig.add_subplot(grids[1, 2]) 
    ax4.errorbar(np.array(x_labels)-10, means[:, 0, 4], yerr=stds[:, 0, 4], fmt='o', linewidth=3, linestyle='dotted', label=method_names[0])
    ax4.errorbar(np.array(x_labels)+10, means[:, 1, 4], yerr=stds[:, 1, 4], fmt='o', linewidth=3, linestyle='dotted', label=method_names[1])
    ax4.set_xticks(x_labels)
    ax4.set_xlabel('sample size')
    ax4.set_ylabel(metric_names[4])
    # ax4.set_ylim(bottom=0)
    # ax4.set_yticks([20, 25, 30, 35, 40])
    ax4.legend(loc='best')


figure_3()


# ================== model-free simulation results ================


paths = os.listdir('./results/model_free')


# 0 'n200_p2_K2_make_circles_factor0.5_units256_bz32_lr1e-3_maxiter5000.npy'
# 1 'n200_p2_K2_make_moons_noise0.05_units256_bz32_lr1e-3_maxiter5000.npy'
# 2 'n200_p2_K3_Anisotropic_units256_bz32_lr5e-3_maxiter5000_scaled.npy'
# 3 'n200_p2_K3_make_blobs_K3_units256_bz32_lr5e-3_maxiter5000.npy'
# 4 'n200_p2_K3_make_class_sep2_units256_bz32_lr5e-3_maxiter5000.npy'
# 5 'n200_p2_K3_varied_variances_units256_bz32_lr1e-3_maxiter6000.npy'


x_labels = [100, 200, 300, 400]
metric_names = ['Adjusted\nRand Index', 'Adjusted\nMutual Information', 
                'Homogeneity', 'Completeness']
method_names = ['ACE', 'GMM', 'K-Means', 'Spectral Clustering', 'Agglomerative Clustering']
data_names = ['Cricles', 'Moons', 'Blobs', 'Anisotropicly distributed blobs', 'Blobs with varied variances', 'Make classification']

idx_list = [0, 1, 3, 2, 5, 4]

def figure_5():
    fig, axes = plt.subplots(2, 3)  # , figsize=(10, 10)
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
            
            axes[i, j].errorbar(np.array(x_labels)-30, means[0, :], yerr=stds[0, :], fmt='o', elinewidth=5, ms=10, label=method_names[0])
            axes[i, j].errorbar(np.array(x_labels)-15, means[1, :], yerr=stds[1, :], fmt='o', elinewidth=5, ms=10, label=method_names[1])
            axes[i, j].errorbar(np.array(x_labels), means[2, :], yerr=stds[2, :], fmt='o', elinewidth=5, ms=10, label=method_names[2])
            axes[i, j].errorbar(np.array(x_labels)+15, means[3, :], yerr=stds[3, :], fmt='o', elinewidth=5, ms=10, label=method_names[3])
            axes[i, j].errorbar(np.array(x_labels)+30, means[4, :], yerr=stds[4, :], fmt='o', elinewidth=5, ms=10, label=method_names[4])
            axes[i, j].set_xticks(x_labels)
            axes[i, j].set_xticklabels(metric_names)
            axes[i, j].set_xlabel('')
            axes[i, j].set_ylabel('')
            axes[i, j].set_ylim(top=1)
            axes[i, j].set_title(data_names[idx], fontdict={'fontsize': 20})
            
            idx += 1
    axes[0, 0].legend(loc='center right')



figure_5()














