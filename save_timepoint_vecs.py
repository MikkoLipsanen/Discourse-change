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

parser = argparse.ArgumentParser(description='Save timepoint vectors')

parser.add_argument('--save_path', type=str, default='results/timepoint_vecs/', help='path to save results')
parser.add_argument('--data_path', type=str, default='data/samples/', help='file containing data')
parser.add_argument('--model_path', type=str, default='model/contrastive_model.pt', help='path to the model')
parser.add_argument('--batch_size', type=int, default=3000, help='defines batch size')
parser.add_argument('--timepoints', type=int, default=100, help='defines number of timepoints in sample set')
parser.add_argument('--ts_size', type=int, default=2, help='defines the size of the test dataset')
parser.add_argument('--lr', type=float, default=0.1, help='defines the learning rate')
parser.add_argument('--hidden_size', type=int, default=10, help='bLSTM hidden state size')
parser.add_argument('--lstm_layers', type=int, default=2, help='number of bLSTM layers')
parser.add_argument('--emb_size', type=int, default=300, help='Size of the document embedding vector')
parser.add_argument('--te_output_size', type=int, default=300, help='Size of the TimepointEmbedding output vector')
parser.add_argument('--rnn_output_size', type=int, default=300, help='Size of the TimepointEmbedding output vector')
parser.add_argument('--lstm_dropout', type=float, default=0.2, help='dropout probability for bLSTM')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--attn_layers', type=int, default=4, help='number of TransformerEncoder layers')
parser.add_argument('--attn_dropout', type=float, default=0.2, help='dropout probability for TransformerEncoder')
parser.add_argument('--job_name', type=str, default='timepoint_vecs', help='name used in the saved result files')

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
ts_sets = test[:args.ts_size]

# Create a custom implementation of Pytorch Dataset class
class ContrDataset(Dataset):
    def __init__(self, files, max_docs, n_samples, d_type):
        self.files = files
        self.unused = files.copy()
        self.ind = 0
        self.max_docs = max_docs
        self.d_type = d_type
        self.n_samples = n_samples
        self.data, self.timepoint_labels, self.pattern_labels = self.get_data()
        
    def get_padding(self, data):
        # Adds padding of zeros to a timepoint if it has less documents than the maximum number in the data
        pad = self.max_docs - data.shape[0]
        data = F.pad(input=data, pad=(0, 0, 0, pad), mode='constant', value=0)
        
        return data
    
    def get_data(self):
        # Selects dataset randomly and then removes it from the file list
        dataset = random.choice(self.unused)
        self.unused.remove(dataset)
        
        # Loads a new dataset defined by the random index
        a_file = open(args.data_path + self.d_type + "/" + dataset, "rb")
        data_dict = pickle.load(a_file)
        a_file.close()
        
        data = data_dict['data']
        timepoint_labels = data_dict['timepoint_labels']
        pattern_label = data_dict['pattern_label']
        pattern_labels = [pattern_label] * len(timepoint_labels)

        # Keeps track of the number of timepoints already loaded
        self.ind += args.timepoints
        
        return data, timepoint_labels, pattern_labels
    
    # Function for reseting the parameters between epochs
    def reset(self):   
        self.ind = 0
        self.unused = self.files.copy()

    def __getitem__(self, index):
        # Loads new dataset when the old one has been used
        if index >= self.ind:
            self.data, self.timepoint_labels, self.pattern_labels = self.get_data()

        # Modifies the index based on the amount of data loaded
        timepoint_index = int(str(index)[-2:])
        
        timepoint = self.data[:,timepoint_index,:]
        timepoint_label = self.timepoint_labels[timepoint_index]
        pattern_label = self.pattern_labels[timepoint_index]
        
        # Adds padding if needed
        if timepoint.shape[0] < self.max_docs:
            timepoint = self.get_padding(timepoint)
            
        return timepoint, timepoint_label, pattern_label
    
    def __len__(self):
        # Returns the total number of timepoints in the data
        n_timepoints = self.n_samples*args.timepoints
    
        return n_timepoints

# Creates test dataset
test_data = ContrDataset(ts_sets, max_docs, len(ts_sets), 'test')

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

# Instantiate the models
tp_emb = TimepointEmbedding(fc_input_size, fc_output_size, args.emb_size, args.n_heads, args.attn_layers, args.attn_dropout).to(device)

# Define optimizer and loss function
optimizer = torch.optim.SGD(list(tp_emb.parameters()) + list(rnn.parameters()), lr=args.lr)

# Keeps count of epochs when training is continued with saved model
epoch_count = 0

print('Loading model from {}'.format(args.model_path))
checkpoint = torch.load(args.model_path)
tp_emb.load_state_dict(checkpoint['tp_emb_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


timepoint_labels = torch.zeros(args.ts_size * args.timepoints)
pattern_labels = torch.zeros(args.ts_size * args.timepoints)
vectors = torch.zeros((args.ts_size * args.timepoints, args.emb_size))

# Function for testing the model
def test_model():

    # Set the networks to evaluation mode (dropout is not applied)
    tp_emb.eval()

    vec_ind = 0
    label_ind = 0

    # Gradients are not calculated during testing
    with torch.no_grad():

        # Loop over test data batches
        for (x_test, y_timepoint, y_pattern) in test_dataloader:

            # Calculate output
            tp_emb_output = tp_emb(x_test)

            n_vecs = tp_emb_output.shape[0]
            vectors[vec_ind:vec_ind + n_vecs, :] = tp_emb_output
            vec_ind += n_vecs

            n_labels = len(y_timepoint)
            timepoint_labels[label_ind:label_ind + n_labels] = y_timepoint
            pattern_labels[label_ind:label_ind + n_labels] = y_pattern
            label_ind += n_labels

test_model()   

results_dict = {'data': vectors, 'timepoint_labels': timepoint_labels, 'pattern_labels': pattern_labels, 'args': args}

file_to_write = open(args.save_path + args.job_name + '.pkl', 'wb')
pickle.dump(results_dict, file_to_write)