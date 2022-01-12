import pickle
import numpy as np
import argparse
import time
import random
from os import listdir
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

# Uses GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

parser = argparse.ArgumentParser(description='Save document vectors')

parser.add_argument('--save_path', type=str, default='results/doc_vecs/', help='path to save results')
parser.add_argument('--data_path', type=str, default='data/samples/', help='file containing data')
parser.add_argument('--model_path', type=str, default='model/contrastive_model.pt', help='path to the model')
parser.add_argument('--batch_size', type=int, default=3000, help='defines batch size')
parser.add_argument('--timepoints', type=int, default=100, help='defines number of timepoints in sample set')
parser.add_argument('--emb_size', type=int, default=300, help='Size of the document embedding vector')
parser.add_argument('--te_output_size', type=int, default=300, help='Size of the TimepointEmbedding output vector')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--attn_layers', type=int, default=4, help='number of TransformerEncoder layers')
parser.add_argument('--attn_dropout', type=float, default=0.2, help='dropout probability for TransformerEncoder')
parser.add_argument('--job_name', type=str, default='doc_vecs', help='name used in the saved result files')
parser.add_argument('--n_docs', type=int, default=2000, help='defines the number of document vectors sampled')
parser.add_argument('--n_samples', type=int, default=5, help='defines the number of datasets sampled')

args = parser.parse_args()
print(args)

# Load the number indicating max amount of docs per timepoint
a_file = open(args.data_path + "tracker_dict", "rb")
tracker_dict = pickle.load(a_file)
a_file.close()
    
max_docs = tracker_dict['max_docs']
print('Max docs: ', max_docs)

# List all files in test folder
test = [f for f in listdir(args.data_path + "test") if isfile(join(args.data_path + "test", f))]

# Shuffle the order of files in the list
random.shuffle(test)

# Select defined number of test files
ts_sets = test[:args.n_samples]

# Create a custom implementation of Pytorch Dataset class
class ClusterDataset(Dataset):
    def __init__(self, files, max_docs, n_samples, d_type):
        self.files = files
        self.unused = files.copy()
        self.ind = 0
        self.max_docs = max_docs
        self.d_type = d_type
        self.n_samples = n_samples
        self.data, self.pattern_labels, self.timepoint_labels, self.category_labels  = self.get_data()
        
    def get_padding(self, data, categories):
        # Adds padding of zeros to a timepoint if it has less documents than the maximum number in the data
        pad = self.max_docs - data.shape[0]
        data = F.pad(input=data, pad=(0, 0, 0, pad), mode='constant', value=0)
        categories += [-1] * pad 
        
        return data, categories
    
    def get_data(self):
        # Selects dataset randomly and then removes it from the file list
        dataset = random.choice(self.unused)
        self.unused.remove(dataset)
        
        # Loads a new dataset defined by the random index
        a_file = open(args.data_path + self.d_type + "/" + dataset, "rb")
        data_dict = pickle.load(a_file)
        a_file.close()
        
        data = data_dict['data']
        pattern_label = data_dict['pattern_label']
        timepoint_labels = data_dict['timepoint_labels']
        pattern_labels = torch.FloatTensor([pattern_label] * len(timepoint_labels))
        category_labels = data_dict['doc_categories']

        # Keeps track of the number of timepoints already loaded
        self.ind += args.timepoints
        
        return data, pattern_labels, timepoint_labels, category_labels

    def __getitem__(self, index):
        # Loads new dataset when the old one has been used
        if index >= self.ind:
            self.data, self.pattern_labels, self.timepoint_labels, self.category_labels = self.get_data()
            
        # Modifies the index based on the amount of data loaded
        timepoint_index = int(str(index)[-2:])
        
        timepoint = self.data[:,timepoint_index,:]
        categories = self.category_labels[timepoint_index]
        
        # Adds padding if needed
        if timepoint.shape[0] < self.max_docs:
            timepoint, categories = self.get_padding(timepoint, categories)
            
        timepoint_label = self.timepoint_labels[timepoint_index]
        timepoint_labels = timepoint_label.expand(timepoint.shape[0])
        pattern_label = self.pattern_labels[timepoint_index]
        pattern_labels = pattern_label.expand(timepoint.shape[0])
        categories = torch.FloatTensor(categories)
            
        return timepoint, timepoint_labels, pattern_labels, categories
    
    def __len__(self):
        # Returns the total number of timepoints in the data
        n_timepoints = self.n_samples*args.timepoints
    
        return n_timepoints

# Creates test dataset
test_data = ClusterDataset(ts_sets, max_docs, len(ts_sets), 'test')

# Creates dataloader for test data
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Define the TimepointEmbedding network class
class TimepointEmbedding(nn.Module):
    def __init__(self, fc_input_size, fc_output_size, features, heads, layers, dropout):
        super(TimepointEmbedding, self).__init__()
        
        # define network layers   
        encoder_layer = nn.TransformerEncoderLayer(d_model=features, nhead=heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers) 
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_size, 200*50)
        self.fc2 = nn.Linear(200*50, fc_output_size)
        self.bn = nn.BatchNorm1d(fc_output_size)


    def forward(self, x):
        # define forward pass
        x = x.permute(1,0,2)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.bn(x)
  
        return x

# Define input and output size for the fully connected layers
fc_input_size = max_docs*args.emb_size
fc_output_size = args.te_output_size

# Instantiate the model
tp_emb = TimepointEmbedding(fc_input_size, fc_output_size, args.emb_size, args.n_heads, args.attn_layers, args.attn_dropout).to(device)


print('Loading model from {}'.format(args.model_path))
checkpoint = torch.load(args.model_path)
tp_emb.load_state_dict(checkpoint['tp_emb_state_dict'])

# Model used for getting document vectors only needs the TransformerEncoder layers
doc_model = nn.Sequential(list(tp_emb.children())[0])

# Function for testing the model
def test_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    doc_model.eval()

    # Number of documents sampled per timepoint
    docs_tp = int(args.n_docs / args.timepoints)

    print('Sampled docs per datapoint: ', docs_tp)

    # Number of timepoints taken to the sample
    tps = int(args.n_samples * args.timepoints) 

    # Tensors for storing the document vectors and labels
    timepoint_labels = torch.zeros((tps, docs_tp))
    pattern_labels = torch.zeros((tps, docs_tp))
    category_labels = torch.zeros((tps, docs_tp))
    vectors = torch.zeros((tps, docs_tp, args.emb_size))

    # Keeps track of the number of timepoints included in the sample
    ind = 0

    # Gradients are not calculated during testing
    with torch.no_grad():

        # Loop over test data batches
        for (x_test, y_timepoint, y_pattern, y_category) in test_dataloader:
 
             # Calculate output
            doc_output = doc_model(x_test)


            for tp in range(x_test.shape[0]):

                if ind < tps:
                    tp_inds = []
                    while len(tp_inds) < docs_tp:
                        # Selects an index for the sample document
                        i = np.random.randint(low=0, high=x_test.shape[1])
                        # Leaves the padding docs out from the sample
                        if (i not in tp_inds) and (y_category[tp,i] != -1):
                            tp_inds.append(i)
                 
                    vectors[ind,:,:] = doc_output[tp,tp_inds,:]
                    timepoint_labels[ind,:] = y_timepoint[tp,tp_inds]
                    pattern_labels[ind,:] = y_pattern[tp,tp_inds]
                    category_labels[ind,:] = y_category[tp,tp_inds]
                    ind += 1
                else:
                    return vectors, category_labels, pattern_labels, timepoint_labels
    
    return vectors, category_labels, pattern_labels, timepoint_labels


vectors, category_labels, pattern_labels, timepoint_labels = test_model()

results_dict = {'data': vectors, 'timepoint_labels': timepoint_labels, 'pattern_labels': pattern_labels, 'category_labels': category_labels, 'args': args}

file_to_write = open(args.save_path + args.job_name + '.pkl', 'wb')
pickle.dump(results_dict, file_to_write)