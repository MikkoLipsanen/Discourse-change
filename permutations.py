import numpy as np
import argparse
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.spatial import distance
from collections import defaultdict

parser = argparse.ArgumentParser(description='Permutation testing')

parser.add_argument('--data_path', type=str, default='results/timepoint_vecs/timepoint_vecs.pkl', help='path to vecs')
parser.add_argument('--n_perm', type=int, default=1000, help='Number of permutations')
parser.add_argument('--n_sets', type=int, default=400, help='Number of samples')

args = parser.parse_args()

# Load the data dict
a_file = open(args.data_path, "rb")
res_dict = pickle.load(a_file)
a_file.close()

# Get vectors, labels and model information
vecs = res_dict['data'].numpy()
labels = res_dict['timepoint_labels'].numpy()
labels = [int(i) for i in labels]

name = res_dict['args'].job_name

print('Total number of timepoint vectors: ', vecs.shape[0])
print('Number of stable timepoints: ', labels.count(0))
print('Number of unstable timepoints: ', labels.count(1))

def get_groups(vecs, labels):
    stable_indices = [i for i, e in enumerate(labels) if e == 0]
    unstable_indices = [i for i, e in enumerate(labels) if e == 1]
    stable_group = vecs[stable_indices,:]
    unstable_group = vecs[unstable_indices,:]
    
    return stable_group, unstable_group, stable_indices, unstable_indices

def sample_inds(stable_indices, unstable_indices):
    
    len_stable = len(stable_indices)
    len_unstable = len(unstable_indices)
    
    sorted_unstable = list(np.sort(unstable_indices))

    stable_inds = []
    unstable_inds = []

    ind = 0
    while ind < args.n_perm:
        rng = np.random.default_rng()
        p = rng.permutation(100)
        sp = list(np.sort(p[:len_stable]))
        su = list(np.sort(p[len_stable:]))
        if su in unstable_inds or su == sorted_unstable:
            continue
        else:
            stable_inds.append(sp)
            unstable_inds.append(su)
            ind += 1
            
    return stable_inds, unstable_inds

def perm_distances(vecs, labels):
    p_distances = []

    stable_group, unstable_group, stable_indices, unstable_indices = get_groups(vecs, labels)
    
    stable_perms, unstable_perms = sample_inds(stable_indices, unstable_indices)
    
    n_stable = stable_group.shape[0]

    stable_mean = np.mean(stable_group, axis=0)
    unstable_mean = np.mean(unstable_group, axis=0)

    gt_dist = distance.cosine(stable_mean, unstable_mean)

    # Define p (number of permutations)
    p = args.n_perm
    pooled = vecs.copy()

    # Permutation loop:
    for i in range(0, p):
        p_1 = pooled[stable_perms[i],:]
        p_2 = pooled[unstable_perms[i],:]
        p_1_mean = np.mean(p_1, axis=0)
        p_2_mean = np.mean(p_2, axis=0)
        p_dist = distance.cosine(p_1_mean, p_2_mean)
        p_distances.append(p_dist)
            
    return gt_dist, p_distances

def plot_results(data, ground_truth, p):
    
    # Plot permutation simulations
    density_plot = sns.histplot(data)

    # Add a line to show the actual difference observed in the data
    density_plot.axvline(
        x=ground_truth, 
        color='red', 
        linestyle='--'
    )

    plt.legend(
        labels=['Mean observed distance: %.4f'%ground_truth, 'Simulated distances'], 
        loc='upper right',
        fontsize=16
    )
    plt.xlabel('Cosine distance', fontsize=16)
    plt.ylabel('Proportion of permutations (log scale)', fontsize=16)
    plt.title('Timepoint vectors: %i sample sets, %i permutations. Mean p-value: %.4f'%(args.n_sets, args.n_perm, p),fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    density_plot.set_yscale("log")

    
    plt.show()

ind = 0
p_ind = 0

gts = np.zeros(args.n_sets)
p_vals = np.zeros(args.n_sets)
perms = np.zeros(args.n_sets * args.n_perm)

j = 0
stable = 0

while j < args.n_sets:
    
    s_vecs = vecs[ind:ind+100,:]
    s_labels = labels[ind:ind+100]
    
    if len(set(s_labels)) > 1:

        gt_dist, p_distances = perm_distances(s_vecs, s_labels)
        gts[j] = gt_dist
        perms[p_ind:p_ind+args.n_perm] = p_distances

        p_val = len(np.where(p_distances >= gt_dist)[0]) / args.n_perm
        p_vals[j] = p_val

        p_ind += args.n_perm
        j += 1
        
    else:
        stable += 1
        
    ind += 100

print('Stable sets: ', stable)    
print('Mean p-value: ', np.mean(p_vals))
print('Min p-value: ', np.min(p_vals))
print('Max p-value: ', np.max(p_vals))

plot_results(perms, np.mean(gts), np.mean(p_vals))