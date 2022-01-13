import numpy as np
import argparse
import pandas as pd
import torch
import ruptures as rpt
from ruptures.metrics import randindex, precision_recall

import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.cluster import adjusted_rand_score

parser = argparse.ArgumentParser(description='Unsupervised timepoint classification and pivot point detection')

parser.add_argument('--data_path', type=str, default='results/timepoint_vecs/timepoint_vecs.pkl', help='path to find data')
parser.add_argument('--res_path', type=str, default='results/', help='path to save results')
parser.add_argument('--n_perm', type=int, default=1000, help='Number of permutations')
parser.add_argument('--win_size', type=int, default=10, help='Size of the smoothing window')
parser.add_argument('--p_threshold', type=float, default=0.05, help='P-value threshold')
parser.add_argument('--emb_size', type=int, default=300, help='Size of the document embedding vector')
parser.add_argument('--timepoints', type=int, default=100, help='Number of timepoints in sample set')
parser.add_argument('--pad_mode', type=str, default='mean', help='What values are used for padding')
parser.add_argument('--dist', type=str, default='cosine', help='Distance measure used for calculating timepoint distances')
parser.add_argument('--vis_samples', type=int, default=0, help='Number of random samples that are visualized')
parser.add_argument('--pr_margin', type=int, default=5, help='Margin parameter for ruptures precision_recall-function')
parser.add_argument('--job_name', type=str, default='unsup_tp_class', help='name used in the saved result files')

args = parser.parse_args()
print(args)

# Load vectors and labels
a_file = open(args.data_path, "rb")
res_dict = pickle.load(a_file)
a_file.close()

vecs = res_dict['data']
labels = res_dict['timepoint_labels']

# Reshape vectors and labels so that the first dimension is the number of sample sets
sample_vecs = torch.reshape(vecs, (-1, args.timepoints, args.emb_size)).detach().numpy()
sample_labels = torch.reshape(labels, (sample_vecs.shape[0], -1)).detach().numpy()

# Adds padding vectors to the beginning and end of the sample set 
# based on the size of the smoothing window
def get_padding(vecs, pad_len):
    # https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    padded_vecs = np.pad(vecs, ((pad_len, pad_len),(0,0)), mode=args.pad_mode)
    
    return padded_vecs

# Calculates pairwse distances between consecutive timepoints in the sample set
def get_pairwise_dist(vecs):
    dists = []
    padded_vecs = np.pad(vecs, ((0, 1),(0,0)), mode=args.pad_mode)
    
    for i in range(vecs.shape[0]):
        if args.dist == 'cosine':
            dist = distance.cosine(padded_vecs[i,:], padded_vecs[i+1,:]) 
        elif args.dist == 'euclidean':
            dist = distance.euclidean(padded_vecs[i,:], padded_vecs[i+1,:]) 

        dists.append(dist)
        
    return dists

# Calculate average distances between mean values of win:size timepoints on both sides of each point
def get_avg_dist(vecs):
    padded_vecs = get_padding(vecs, args.win_size)
    avg_dists = []
    
    for i in range(vecs.shape[0]):
        j = i+10
        avg_dist = distance.cosine(np.mean(padded_vecs[j-args.win_size:j,:], axis=0), np.mean(padded_vecs[j:j+args.win_size,:], axis=0))
        avg_dists.append(avg_dist)
        
    return avg_dists

# Plots the pairwise distances 
def plot_pairwise(dists, labels):
    
    pivots = np.where(labels[:-1] != labels[1:])[0] + 1
    start_points = pivots[::2]
    end_points = pivots[1::2]
        
    plt.plot(dists, label='Pairwise dist')
    for i in range(len(start_points)):
        plt.axvline(x=start_points[i], ymin=0, ymax=1, c='r', linestyle='--',label='Unstable pattern starts')
        plt.axvline(x=end_points[i], ymin=0, ymax=1, c='g', linestyle='--',label='Unstable pattern ends')
        
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())
    plt.title('Pairwise cosine distances')

    plt.show()

# Plots the timeseries of p-values from the permutations
def plot_p(p_vals, labels):
    
    pivots = np.where(labels[:-1] != labels[1:])[0] + 1
    end_points = pivots[1::2]
    start_points = pivots[::2]
        
    plt.plot(p_vals, label='P-values')
    for i in range(len(start_points)):
        plt.axvline(x=start_points[i], ymin=0, ymax=1, c='r', linestyle='--',label='Unstable pattern starts')
        plt.axvline(x=end_points[i], ymin=0, ymax=1, c='g', linestyle='--',label='Unstable pattern ends')
        
    plt.axhline(y=args.p_threshold, xmin=0, xmax=len(labels), c='m', linestyle='--',label='Threshold: {}'.format(args.p_threshold))  
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())
    plt.title('P-values based on averaged cosine distances from {} permutations'.format(args.n_perm))

    plt.show()

# Plot random samples from the data
def plot_samples(n_samples, p_values, sample_vecs, sample_labels):

    ind = np.random.randint(sample_labels.shape[0], size=n_samples)
    for i in ind:
        vecs = sample_vecs[i,:,:]
        labels = sample_labels[i,:]
        p_vals = p_values[i,:]
        pairwise_dist = get_pairwise_dist(vecs)

        plot_pairwise(pairwise_dist, labels)
        plot_p(p_vals, labels)


# Calculates the permutation distances and p-values, which are used with the threshold to predict the timepoint labels
def permutations(sample_vecs, sample_labels):
    
    correct = 0
    prec = 0
    rec = 0
    fb = 0
    rand_score = 0
    
    pred_labels = np.zeros(sample_labels.shape)
    p_values = np.zeros(sample_labels.shape)
    gt_dists = np.zeros(sample_labels.shape)

    n_samples = sample_vecs.shape[0]
    
    for i in range(n_samples):
        vecs = sample_vecs[i,:,:]
        labels = sample_labels[i,:]
        
        # Get ground truth avg distances
        gt_dist = get_avg_dist(vecs)
        gt_dists[i,:] = gt_dist
        
        # Define p (number of permutations)
        p = args.n_perm
        pooled = vecs.copy()
    
        # Array for the sample distances
        perm_dists = np.zeros((p, len(gt_dist)))
        # Array for the sample p-values
        p_vals = np.zeros(len(gt_dist))

        # Permutation loop:
        for j in range(p):
            # Shuffle the data
            np.random.shuffle(pooled)
            # Get paddings
            dist = get_avg_dist(pooled)
            perm_dists[j,:] = dist

        for k in range(len(p_vals)):
            avg_dists = perm_dists[:,k]
            # Ground truth distance at timepoint j is compared to all permutation distances at the same point
            p_val = len(np.where(avg_dists >= gt_dist[k])[0]) / p
            p_vals[k] = p_val
        
        # Predicts the labels so that all p-values under the threshold get label 1 (unstable) 
        # and all others get label 0 (stable)
        y_hat = (p_vals < args.p_threshold).astype(int)
        # Correct predictions per sample set
        correct += (labels == y_hat).sum()
        pr, re, f, s = precision_recall_fscore_support(labels, y_hat, average='binary', zero_division=0)
        rand = adjusted_rand_score(labels, y_hat)
        
        prec += pr
        rec += re
        fb += f
        rand_score += rand

        p_values[i,:] = p_vals
        pred_labels[i,:] = y_hat

    # Total number of timepoints in input data
    timepoints = sample_labels.shape[0] * sample_labels.shape[1] 

    # Accuracy = total number of correct predictions / total number of timepoints
    accuracy = correct / timepoints
    prec_mean = prec / n_samples
    rec_mean = rec / n_samples
    fb_mean = fb / n_samples
    rand_mean = rand_score / n_samples
    
    return accuracy, prec_mean, rec_mean, fb_mean, pred_labels, p_values, gt_dists, rand_mean


accuracy, prec_mean, rec_mean, fb_mean, pred_labels, p_values, dists, rand_mean = permutations(sample_vecs, sample_labels)


print("Accuracy of label predictions based on the p-values: %.3f"%accuracy)
print("Mean F-beta score of label predictions based on the p-values: %.3f"%fb_mean)
print("Mean precision of label predictions based on the p-values: %.3f"%prec_mean)
print("Mean recall of label predictions based on the p-values: %.3f"%rec_mean)
print("Adjusted rand score of label predictions based on the p-values: %.3f"%rand_mean)

if args.vis_samples > 0:
    plot_samples(args.vis_samples, p_values, sample_vecs, sample_labels)

############################################################################
## Ruptures
############################################################################

def get_rupture_pivots(freqs, labels):
  
    algo = rpt.Window(width=5, model="normal").fit(freqs) #"l2", "l1", "rbf", "linear", "normal", "ar"
    pred_pivots = algo.predict(pen=0.5)
    
    true_pivots = list(np.where(labels[:-1] != labels[1:])[0] + 1)
    
    #rpt.show.display(freqs, true_pivots, pred_pivots, figsize=(10, 6))
    #plt.show()

    return pred_pivots, true_pivots

def rupture_dists(dists, labels):
    n_samples = dists.shape[0]
    rand_scores = []
    precs = []
    recalls = []
    skipped = 0

    for i in range(n_samples):
        pred_pivots, true_pivots = get_rupture_pivots(dists[i,:], labels[i,:])
        true_pivots.append(100)
        rand = randindex(true_pivots, pred_pivots)
        if len(true_pivots) > 1:
            prec, recall = precision_recall(true_pivots, pred_pivots, margin=args.pr_margin)
            precs.append(prec)
            recalls.append(recall)
        else:
            skipped += 1

        rand_scores.append(rand)
        
    print('Skipped', skipped)

    return np.mean(rand_scores), np.mean(precs), np.mean(recalls)

dist_rand, dist_prec, dist_rec = rupture_dists(dists, sample_labels)

p_rand, p_prec, p_rec = rupture_dists(p_values, sample_labels)

print("Rand score of ruptures pivot point detection based on the distance timeline: %.3f"%dist_rand)
print("Precision of ruptures pivot point detection based on the distance timeline: %.3f"%dist_prec)
print("Recall score of ruptures pivot point detection based on the distance timeline: %.3f"%dist_rec)

print("Rand score of ruptures pivot point detection based on the p-value timeline: %.3f"%p_rand)
print("Precision of ruptures pivot point detection based on the p-value timeline: %.3f"%p_prec)
print("Recall score of ruptures pivot point detection based on the p-value timeline: %.3f"%p_rec)

results = {'accuracy': accuracy, 'f_score': fb_mean, 'prec_p': prec_mean, 'rec_p': rec_mean, 'rand_score': rand_mean,
        'dist_rand': dist_rand, 'dist_prec': dist_prec, 'dist_rec': dist_rec, 'p_rand': p_rand,
        'p_prec': p_prec, 'p_rec': p_rec, 'args': args}

# Save loss and accuracy
file_to_write = open(args.res_path + args.job_name + '.pkl', 'wb')
pickle.dump(results, file_to_write)