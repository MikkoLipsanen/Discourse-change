import numpy as np
import argparse
import time
import pandas as pd

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pickle5 as pickle
import plotly.express as px


parser = argparse.ArgumentParser(description='t-SNE')

parser.add_argument('--data_path', type=str, default='results/timepoint_vecs/timepoint_vecs.pkl', help='path to vecs')
parser.add_argument('--perplexity', type=int, default=30, help='perplexity')
parser.add_argument('--n_iter', type=int, default=300, help='number of iterations')
parser.add_argument('--n_vecs', type=int, default=50000, help='number of vectors used')

args = parser.parse_args()
print(args)

a_file = open(args.data_path, "rb")
res_dict = pickle.load(a_file)
a_file.close()

vecs = res_dict['data'][:args.n_vecs,:]
pt_labels = res_dict['pattern_labels'].tolist()[:args.n_vecs]
tp_labels = res_dict['timepoint_labels'].tolist()[:args.n_vecs]
name = res_dict['args'].job_name

pattern_labels = {1: 'Up', 2: 'Down', 3: 'Up-Down', 4: 'Down-Up', 5: 'Spike Up', 6: 'Spike Down', 7: 'Stable'}
y_pt = [pattern_labels[i] for i in pt_labels]
color_pt = {'Up':'blue','Down':'red','Up-Down':'orange','Down-Up': 'green','Spike Up': 'dodgerblue', 'Spike Down':'brown', 'Stable':'black'}

timepoint_labels = {0: 'Stable', 1: 'Unstable'}
y_tp = [timepoint_labels[int(i)] for i in tp_labels]
color_tp = {'Stable':'blue','Unstable':'red'}

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=args.perplexity, n_iter=args.n_iter)
tsne_result = tsne.fit_transform(vecs)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


tsne_result_pt = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y_pt})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_pt, ax=ax,s=120, palette=color_pt, alpha=0.7)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_xlabel("Dimension 1",fontsize=16)
ax.set_ylabel("Dimension 2",fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, prop={"size":16})
plt.title('Pattern t-SNE, {} timepoints, perplexity {}'.format(vecs.shape[0], args.perplexity), fontsize=20)
 

plt.show()

tsne_result_tp = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y_tp})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_tp, ax=ax,s=120, palette=color_tp, alpha=0.7)
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_xlabel("Dimension 1",fontsize=16)
ax.set_ylabel("Dimension 2",fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, prop={"size":16})
plt.title('Timepoint t-SNE, {} timepoints, perplexity {}'.format(vecs.shape[0], args.perplexity), fontsize=20)

plt.show()