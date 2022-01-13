import numpy as np
import argparse
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import rand_score, adjusted_rand_score
import matplotlib.pyplot as plt
import pickle


parser = argparse.ArgumentParser(description='K-means clustering')

file_name = 'Noise0001_vecs.pkl'

parser.add_argument('--data_path', type=str, default='results/timepoint_vecs/timepoint_vecs.pkl', help='path to vecs')
parser.add_argument('--n_samples', type=int, default=2, help='number of samples')

args = parser.parse_args()
print(args)

a_file = open(args.data_path, "rb")
res_dict = pickle.load(a_file)
a_file.close()

vecs = res_dict['data']
pt_vecs = vecs[:args.n_samples * 100,:].numpy()

n_samples = int(vecs.shape[0] / 100)
tp_vecs = torch.reshape(vecs,(n_samples, -1, 300)).numpy()[:args.n_samples,:,:]

tp_labels = res_dict['timepoint_labels']
tp_labels = torch.reshape(tp_labels,(n_samples, -1)).numpy()[:args.n_samples,:]
pt_labels = res_dict['pattern_labels'][:args.n_samples * 100].numpy()
print(tp_labels.shape)
n_pt = len(np.unique(pt_labels))
name = res_dict['args'].job_name

tp_rands = []

for i in range(args.n_samples):
    sample_labels = tp_labels[i,:]
    sample_vecs = tp_vecs[i,:,:]
    n_tp = len(np.unique(sample_labels))
    if n_tp > 1:
        tp_kmeans= KMeans(n_clusters=n_tp, random_state=2)
        tp_preds = tp_kmeans.fit_predict(sample_vecs)
        tp_rand = adjusted_rand_score(sample_labels, tp_preds)
        tp_rands.append(tp_rand)

pt_kmeans= KMeans(n_clusters=n_pt, random_state=4)
pt_preds = pt_kmeans.fit_predict(pt_vecs)
pt_rand = adjusted_rand_score(pt_labels, pt_preds)

tp_mean = np.mean(tp_rands)
tp_min = min(tp_rands)
tp_max = max(tp_rands)

print('Mean adjusted rand score for K-means binary clustering of timepoint vectors: %.3f'%tp_mean)
print('Min adjusted rand score for K-means binary clustering of timepoint vectors: %.3f'%tp_min)
print('Max adjusted rand score for K-means binary clustering of timepoint vectors: %.3f'%tp_max)

print('Adjusted rand score for K-means pattern based clustering of timepoint vectors: %.3f'%pt_rand)