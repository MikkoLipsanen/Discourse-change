#import pickle
import pickle
import numpy as np
import pandas as pd
import argparse
from scipy.stats import norm
import matplotlib.pyplot as plt
import random
import itertools
from collections import Counter, defaultdict
from sklearn import preprocessing
import torch
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser(description='Create synthetic datasets')

parser.add_argument('--data_path', type=str, default='data/df_light.pkl', help='file containing data')
parser.add_argument('--emb_path', type=str, default='data/embeddings_dict.pkl', help='file containing the embeddings')
parser.add_argument('--save_path', type=str, default='data/samples/', help='path to save results')
parser.add_argument('--tr_samples', type=int, default=5000, help='Number of training datasets created and used')
parser.add_argument('--ts_samples', type=int, default=2000, help='Number of test datasets created and used')
parser.add_argument('--num_timepoints', type=int, default=100, help='number of timepoints in each sample')
parser.add_argument('--min_docs', type=int, default=4000, help='minimum number of documents in each datapoint')
parser.add_argument('--max_docs', type=int, default=5000, help='maximum number of documents in each datapoint')
parser.add_argument('--test_categories', type=list, default=['autot', 'musiikki', 'luonto', 'vaalit', 'taudit'], help='News categories used in the test data')
parser.add_argument('--tr_val_categories', type=list, default=['työllisyys','jääkiekko', 'kulttuuri', 'rikokset', 'koulut', 'tulipalot', 'ruoat'], help='News categories used in the training and validation data')
parser.add_argument('--n_topics_tr', type=int, default=7, help='number of topics used in the training samples')
parser.add_argument('--n_topics_ts', type=int, default=5, help='number of topics used in the test samples')
parser.add_argument('--n_unstable', type=int, default=1, help='number of unstable categories used in the samples')
parser.add_argument('--n_stable', type=int, default=1, help='number of stable categories used in the samples')
parser.add_argument('--add_noise', type=str, default='no', help='defines whether random noise is added to the discourse patterns')
parser.add_argument('--noise_std', type=float, default=0.0001, help='defines the std of the gaussian random noise')


parser.add_argument('--rand_docs', type=str, default='no', help='defines whether random documents are selected to the sample')

args = parser.parse_args('')
print(args)

############Load data##############

# Load the dataset with article ids and topics
with open(args.data_path, 'rb') as handle:
    df = pickle.load(handle)

# Load the embedding data in dictionary form
with open(args.emb_path, 'rb') as handle:
    embeddings_dict = pickle.load(handle)


######Split data based on topics#####

# Split dataset into 2 sets based on the given categories used in the test data
# Train and validation datasets are sampled from articles that don't contain any of the test categories
def split_dataset(df, cats):

    mask = df['subjects'].apply(lambda x: any(item for item in cats if item in x))
    tr_val_data = df[~mask].copy()
    test_data = df[mask].copy()

    return tr_val_data, test_data

# Selects one category 'label' for each article
def extract_categories(df, cats):
    clusters = []
    
    for cat in cats:
        is_cat = df['subjects'].apply(lambda x: (cat in x))
        df_filtered = df[is_cat].copy()
        df_filtered['category'] = cat
        clusters.append(df_filtered)
        
    df_merged = pd.concat(clusters, ignore_index=True)
    df_merged = df_merged.drop(columns=['subjects'])
    
    return df_merged


# Split dataset into two parts so that articles in train and validation sets don't contain categories chosen for the test set
tr_val_data, test_data = split_dataset(df, args.test_categories)

# Select one "representative" category for each article 
tr_val_df = extract_categories(tr_val_data, args.tr_val_categories)
test_df = extract_categories(test_data, args.test_categories)

del tr_val_data
del test_data

# Create a dataset for selecting random samples
if args.rand_docs == 'yes':    
    # Create a list of all topics in the dataset
    subject_list = list(itertools.chain(*list(df['subjects'])))

    # Count the occurrence of each topic (number of articles with the topic)
    subject_count = Counter(subject_list)

    # Select categories for training and validation that are not in test categories and are not extremely rare
    rand_cats = [x[0] for x in subject_count.most_common() if x[1] > 8000 and x[0] not in args.test_categories]

    del subject_list
    del subject_count
    
    rand_df = extract_categories(df, rand_cats)
    
    le = preprocessing.LabelEncoder()
    le.fit(rand_df['category'])
    y_rand = le.transform(rand_df['category'])
    rand_df['category_no'] = y_rand

# Create numeric labels for categories
le = preprocessing.LabelEncoder()
le.fit(tr_val_df['category'])
y = le.transform(tr_val_df['category'])
tr_val_df['category_no'] = y
le.fit(test_df['category'])
y_test = le.transform(test_df['category'])
test_df['category_no'] = y_test

#############Sample datasets###################

# Sampling patterns for the synthetic data
def linear_pattern(n=1, start=0, stop=100, change_rate=1):
    """
    Sampling up pattern, start and end in random points
    """
    x = np.arange(start, stop)
    # normalize x to range 0-1
    y = (x - start) / (stop - start)
    freq_rates = n + y * n * change_rate
    
    return freq_rates

def sigmoid_pattern(n=1, start=0, stop=100, change_rate=1):
    x = np.arange(start, stop)
    mid = int((stop - start) / 2)
    y = 1 / (1 + np.exp(-0.1* (x-mid) ))
    y = (y - y.min()) / (y.max() - y.min())
    
    freq_rates = n + y * n * change_rate
    return freq_rates

def flat_pattern(n=1, start=0, stop=100):
    freq_rates = np.ones(stop-start) * n
    return freq_rates

def bell_pattern(n=1, start=0, stop=100, change_rate=1, std=0):
    sample_list = []
    time_range = stop - start
    
    x = np.arange(start, stop)
    mu = int(time_range / 2)
    
    std = std if std else int(time_range / 5)
    y = norm.pdf(np.arange(time_range), mu, std)
    # scale 0-1
    y = (y - y.min()) / (y.max() - y.min())
    # add n docs
    freq_rates = n + y * n * change_rate
    
    return freq_rates


def sample_pattern(pattern, timeline=100, change_rate=0.01):
    sample = None
    
    if pattern == 'up':
        lower_p = np.random.randint(low=1, high=timeline-30)
        upper_p = np.random.randint(low=lower_p+20, high=timeline)
        
        # f1, f2, f3 [-1] is the start of freqs ratio for the pattern as the chaning variable
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = sigmoid_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        change_points = np.array([lower_p, upper_p])
        
    elif pattern == 'down':
        lower_p = np.random.randint(low=1, high=timeline-30)
        upper_p = np.random.randint(low=lower_p+20, high=timeline)
        
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = sigmoid_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=-change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        change_points = np.array([lower_p, upper_p])
        
    elif pattern == 'spike_up':
        n_point = np.random.randint(1, 5)
        invalid = True
        
        while invalid:
            change_points = np.sort(np.random.choice(range(5, timeline - 5), n_point, replace=False))
            diff = np.diff(change_points)
            invalid = len(np.where(diff < 10)[0])
            
        change_rates = np.random.uniform(0.3, change_rate, n_point)
        cur_p = 0
        cur_n = 1
        
        time_freqs = []
        
        for i, p in enumerate(change_points):
            #print(cur_p, p - 2)
            f1 = flat_pattern(cur_n, start=cur_p, stop=p-2)
            cur_n = f1[-1]
            f2 = bell_pattern(cur_n, start=p-2, stop=p+3, change_rate=change_rates[i], std=0.1)
            cur_n = f2[-1]
            
            time_freqs.append(f1)
            time_freqs.append(f2)
            
            cur_p = p + 3
            
            if i == len(change_points) - 1:
                f3 = flat_pattern(cur_n, start=cur_p, stop=timeline)
                time_freqs.append(f3)

        time_freqs = np.concatenate(time_freqs)
        time_freqs = time_freqs / time_freqs.sum()
        
    elif pattern == 'spike_down':
        n_point = np.random.randint(1, 5)
        invalid = True
        # generate n points with min distance 10
        while invalid:
            change_points = np.sort(np.random.choice(range(5, timeline - 5), n_point, replace=False))
            diff = np.diff(change_points)
            invalid = len(np.where(diff < 10)[0])
            
        change_rates = np.random.uniform(0.3, change_rate, n_point)
        cur_p = 0
        cur_n = 1
        
        time_freqs = []
        
        for i, p in enumerate(change_points):

            f1 = flat_pattern(cur_n, start=cur_p, stop=p-2)
            cur_n = f1[-1]
            f2 = bell_pattern(cur_n, start=p-2, stop=p+3, change_rate=-change_rates[i], std=0.1)
            cur_n = f2[-1]
            
            time_freqs.append(f1)
            time_freqs.append(f2)
            
            cur_p = p + 3
            
            if i == len(change_points) - 1:
                f3 = flat_pattern(cur_n, start=cur_p, stop=timeline)
                time_freqs.append(f3)
            
        time_freqs = np.concatenate(time_freqs)
        time_freqs = time_freqs / time_freqs.sum()
        
    elif pattern == 'up_down':
        lower_p = np.random.randint(low=1, high=timeline-20)
        upper_p = np.random.randint(low=lower_p+10, high=timeline)
        
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = bell_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()
        
        mid_p = int(lower_p + (upper_p - lower_p) / 2)
        change_points = np.array([lower_p, mid_p, upper_p])
        
    elif pattern == 'down_up':
        lower_p = np.random.randint(low=1, high=timeline-20)
        upper_p = np.random.randint(low=lower_p+10, high=timeline)
        
        f1 = flat_pattern(1, start=0, stop=lower_p)
        f2 = bell_pattern(f1[-1], start=lower_p, stop=upper_p, change_rate=-change_rate)
        f3 = flat_pattern(f2[-1], start=upper_p, stop=timeline)
        
        # the frequency ratio 
        
        time_freqs = np.concatenate((f1, f2, f3))
        time_freqs = time_freqs / time_freqs.sum()

        mid_p = int(lower_p + (upper_p - lower_p) / 2)
        change_points = np.array([lower_p, mid_p, upper_p])
        
    else:
        time_freqs = flat_pattern(1, start=0, stop=timeline)
        time_freqs = time_freqs / time_freqs.sum()
        change_points = np.empty(shape=(0,))
        
    return time_freqs, change_points.astype(int)

# Creates samples
def get_sample_sets(df, d_type='train', n_samples=100, min_doc=50, max_doc=100, timeline=100, change_rates=[0.5, 1]):
    
    unique_categories = list(df['category'].unique())

    if d_type == 'test':
        categories = random.sample(unique_categories, args.n_topics_ts)
    elif d_type == 'train':
        categories = random.sample(unique_categories, args.n_topics_tr)
    else:
        print("Select one of the following sample data types: 'test', 'train'")
    
    samples = []  
    labels = [] 

    tracker = pd.DataFrame(columns=['unstable_categories', 'stable_categories', 'patterns', 'pivots'])
    patterns_labels = [('up',1), ('down',2), ('up_down',3), ('down_up',4), ('spike_up',5), ('spike_down',6), ('stable',7)]
    
    g = df.groupby(['category'])
    
    for _ in range(n_samples):

        n_cats = args.n_unstable + args.n_stable
        cats = np.random.choice(categories, n_cats, replace=False)
        unstable_cats = cats[:args.n_unstable]
        stable_cats = cats[args.n_unstable:]

        if args.rand_docs == 'yes':
            df_random = rand_df.copy()
            df_random = df_random.loc[~df_random['category'].isin(cats)]
            cats = np.append(cats,'random')

        indices = np.random.choice(len(patterns_labels), args.n_unstable, replace=False)
        sample_patterns_labels = np.array(patterns_labels)[indices]

        sample_patterns = [pl[0] for pl in sample_patterns_labels]
        sample_labels = [pl[1] for pl in sample_patterns_labels]
        sample_labels.sort()

        labels.append(int(''.join(sample_labels)))

        sample_change_rates = np.random.uniform(*change_rates, size=args.n_unstable)
        ind = 0

        pivot_points = []
        df_sample = []
        
        for c in cats:
            if c in unstable_cats:
                freqs, pivots = sample_pattern(sample_patterns[ind], timeline=timeline, change_rate=sample_change_rates[ind])
                pivot_points.append(pivots)
                ind += 1
            else:
                freqs, _ = sample_pattern('stable', timeline=timeline)
            
            # Adds noise to the discourse patterns
            if args.add_noise == 'yes':
                freqs = np.random.normal(0, args.noise_std, freqs.shape) + freqs
        
            # get n_doc, which is random between min and max but not exceed the total docs in cluster
            n_doc = np.random.randint(min_doc, max_doc)

            if c != 'random':
                df_cat = g.get_group(c)[['id', 'category', 'category_no']]
                df_len = len(df_cat)
                n_doc = min(n_doc, df_len)
                docs_num = (n_doc * freqs).astype(int)
                sample = df_cat.sample(n_doc)
                       
            else:
                freqs = np.random.normal(0, args.noise_std, freqs.shape) + freqs
                n_doc = min(n_doc, len(df_random))
                docs_num = (n_doc * freqs).astype(int)
                sample = df_random.sample(n_doc)
    
            sample = sample[['id', 'category', 'category_no']]
            sample['time'] = -1
            
            # assign the sampled time points to the docs
            cur = 0
            for i, n in enumerate(docs_num):
                sample.iloc[cur:cur+n, sample.columns.get_loc("time")] = i
                cur += n

            # because the freq is converted to int, so the n_doc > docs_num. so some articles will remain -1 for time, we need to prunt those
            sample = sample[sample['time'] > -1]
            df_sample.append(sample)
            
        df_sample = pd.concat(df_sample, ignore_index=True)
        #df_sample = df_sample.sample(frac=frac)
        samples.append(df_sample)
        
        tracker = tracker.append({'unstable_categories':unstable_cats, 'stable_categories':stable_cats, 'patterns': sample_patterns, 'pivots': pivot_points}, ignore_index=True)
        
    return samples, labels, tracker

def visualize_trending(data):
    
    for df in data:
        fig, ax = plt.subplots(figsize=(20, 10))
        
        for name, group in df.groupby(['category']):
            g = group.groupby(['time'])['id'].count()
            ax.plot(g.index, g.values, label=name)
            
        ax.set(xlabel='time', ylabel='Numbers')
        ax.legend()
        ax.grid()

    plt.show()


# Creates the labels for the timepoints based on the pivot points
def convert_pivots(tracker, timepoints):
    
    df = tracker.copy()
    df['labels'] = pd.Series(np.array)
    
    for j, row in df.iterrows():

        row['labels'] = np.zeros(timepoints).astype(int)
        sample_pivots = []
        
        for i in range(len(row['patterns'])):
            
            pivots = []
            if row['patterns'][i] == 'up_down' or row['patterns'][i] == 'down_up':
                pivots = [row['pivots'][i][0], row['pivots'][i][2]]
                row['labels'][pivots[0]:pivots[1]] = 1

            elif row['patterns'][i] == 'spike_up' or row['patterns'][i] == 'spike_down':
                for p in row['pivots'][i]:
                    pivots.append(p-2)
                    pivots.append(p+2)

                for i in range(0,len(pivots)-1, 2):
                    row['labels'][pivots[i]:pivots[i+1]] = 1   
                    
            elif row['patterns'][i] == 'up' or row['patterns'][i] == 'down':
                pivots = [row['pivots'][i][0],row['pivots'][i][1]]
                row['labels'][pivots[0]:pivots[1]] = 1
                
            sample_pivots += pivots
            
        row['pivots'] = np.array(sample_pivots)
               
    return df

# Adds the labels to the sample dataframes
def add_timepoint_labels(test_samples, tracker_labels):

    samples = []

    for i, df in enumerate(test_samples):
        data = df.copy()
        labels = []
        for j, row in data.iterrows():
            timepoint = row['time']
            labels.append(tracker_labels['labels'][i][timepoint])
            
        data['label'] = labels
        samples.append(data)
        
    return samples

# Create defined number of samples using categories based on the chosen type (train, validation, test)
def get_samples(d_type, n_samples):
    
    if d_type == 'test':
        df = test_df
    else:
        df = tr_val_df
        
    samples, pattern_labels, tracker = get_sample_sets(df, d_type=d_type, n_samples=n_samples, timeline=args.num_timepoints, min_doc=args.min_docs, max_doc=args.max_docs, change_rates=[0.5, 1])
   
    #visualize_trending(samples)

    tracker_labels = convert_pivots(tracker, args.num_timepoints)
    
    samples_labels = add_timepoint_labels(samples, tracker_labels)
    
    return samples_labels, pattern_labels, tracker_labels

# Gets the embeddings corresponding to the articles
def get_embeddings(dataframe):
    
    df = dataframe.copy()

    embeddings = {}
    labels = {}
    cats = defaultdict(list)

    tensor_list = []
    label_list = []
    cat_list = []
    missing = 0

    for i, row in df.iterrows():
        if row['id'] in embeddings_dict:
            emb = embeddings_dict[row['id']]
            timepoint = row['time']
            label = row['label']
            cat = row['category_no']
            cats[timepoint].append(cat)

            if timepoint in embeddings:
                embeddings[timepoint] = np.vstack([embeddings[timepoint], emb])
            else:
                embeddings[timepoint] = emb

            if timepoint not in labels:
                labels[timepoint] = label

        else:
            missing += 1
        
    print('Missing {} embeddings'.format(missing))
           
    for t in range(len(embeddings)):
        tensor = torch.from_numpy(embeddings[t])
        tensor_list.append(tensor)
        label_list.append(labels[t])
        cat_list.append(cats[t])

    # Adds padding to tensors that have less documents than the maximum number of docs in timepoint
    padded_data = pad_sequence(tensor_list)
    
    # Add category labels for the padding 'docs'
    for t in range(padded_data.shape[1]):
        if len(cat_list[t]) < padded_data.shape[0]:
            n_pads = padded_data.shape[0] - len(cat_list[t])
            cat_list[t] += [-1] * n_pads

    return padded_data, label_list, cat_list


# Creates a dict based on the data sample where timepoints are keys and embeddings are the values
def get_data(d_type, n_samples):
    
    samples, pattern_labels, tracker = get_samples(d_type, n_samples)
    
    # Variable for recording the maximum amount of documents per timepoint for padding purposes
    max_docs = 0
    
    for ind, dataframe in enumerate(samples):
        data, timepoint_labels, cats = get_embeddings(dataframe)
        
        if data.shape[0] > max_docs:
            max_docs = data.shape[0]
  
        # Saves each sample as a dict with data tensor and labels tensor
        sample_dict = {'data': data, 'doc_categories': cats, 'pattern_label': pattern_labels[ind], 'timepoint_labels': torch.FloatTensor(timepoint_labels)}
        output_file = open(args.save_path + d_type + "/%i.pkl" %ind, "wb")
        pickle.dump(sample_dict, output_file)
        output_file.close()      
              
    return max_docs, tracker, pattern_labels

max_docs = 0

# Get test data
max_docs_ts, ts_tracker, ts_labels = get_data('test', args.ts_samples)

del test_df

# Get training data
max_docs_tr, tr_tracker, tr_labels = get_data('train', args.tr_samples)

# get the maximum number of docs per timepoint in all data
if max_docs_ts > max_docs_tr:
    max_docs = max_docs_ts
else:
    max_docs = max_docs_tr

print("Max size: ", max_docs)

# Save tracker and max number of docs to be used in padding
tracker_dict = {'train_tracker': tr_tracker, 'test_tracker': ts_tracker, 'max_docs': max_docs}
output_file = open(args.save_path + "tracker_dict", "wb")
pickle.dump(tracker_dict, output_file)
output_file.close() 