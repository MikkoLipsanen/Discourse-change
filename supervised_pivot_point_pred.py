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

from ruptures.metrics import randindex, precision_recall

parser = argparse.ArgumentParser(description='Supervised pivot point detection')

parser.add_argument('--data_path', type=str, default='data/samples/', help='file containing data')
parser.add_argument('--res_path', type=str, default='results/', help='path to save results')
parser.add_argument('--model_path', type=str, default='model/supervised_tp_model.pt', help='path to save the model')
parser.add_argument('--batch_size', type=int, default=100, help='defines batch size')
parser.add_argument('--timepoints', type=int, default=100, help='defines number of timepoints in sample set')
parser.add_argument('--ts_size', type=int, default=1000, help='defines the size of the test dataset')
parser.add_argument('--lr', type=float, default=0.01, help='defines the learning rate')
parser.add_argument('--hidden_size', type=int, default=10, help='bLSTM hidden state size')
parser.add_argument('--lstm_layers', type=int, default=2, help='number of bLSTM layers')
parser.add_argument('--emb_size', type=int, default=300, help='Size of the document embedding vector')
parser.add_argument('--te_output_size', type=int, default=300, help='Size of the TimepointEmbedding output vector')
parser.add_argument('--rnn_output_size', type=int, default=300, help='Size of the TimepointEmbedding output vector')
parser.add_argument('--lstm_dropout', type=float, default=0.2, help='dropout probability for bLSTM')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--attn_layers', type=int, default=4, help='number of TransformerEncoder layers')
parser.add_argument('--attn_dropout', type=float, default=0.2, help='dropout probability for TransformerEncoder')
parser.add_argument('--smoothing_type', type=str, default='no', help='Type of smoothing calculation')
parser.add_argument('--window_size', type=int, default=9, help='Size of the smoothing window')
parser.add_argument('--pr_margin', type=int, default=5, help='Margin parameter for ruptures precision_recall-function')
parser.add_argument('--job_name', type=str, default='supervised_pivot_point_pred', help='name used in the saved result files')

args = parser.parse_args()
print(args)

# Uses GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

############Get test data#########

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

#########Initialize dataset and dataloader#######

# Create a custom implementation of Pytorch Dataset class
class ClassificationDataset(Dataset):
    def __init__(self, files, max_docs, n_samples, d_type):
        self.files = files
        self.unused = files.copy()
        self.ind = 0
        self.max_docs = max_docs
        self.d_type = d_type
        self.n_samples = n_samples
        self.data, self.pattern_label, self.timepoint_labels = self.get_data()
        self.batch_pattern_labels = [self.pattern_label]
        
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
        pattern_label = data_dict['pattern_label']
        timepoint_labels = data_dict['timepoint_labels']

        # Keeps track of the number of timepoints already loaded
        self.ind += args.timepoints
        
        return data, pattern_label, timepoint_labels

    # Function that allows reseting the parameters between epochs
    def reset(self):   
        self.ind = 0
        self.unused = self.files.copy()

    def reset_batch_labels(self):
        self.batch_labels = []

    def __getitem__(self, index):
        # Loads new dataset when the old one has been used
        if index >= self.ind:
            self.data, self.pattern_label, self.timepoint_labels = self.get_data()
            self.batch_pattern_labels.append(self.pattern_label)
        
        # Modifies the index based on the amount of data loaded
        timepoint_index = int(str(index)[-2:])

        timepoint = self.data[:,timepoint_index,:]
        label = self.timepoint_labels[timepoint_index]
        
        # Adds padding if needed
        if timepoint.shape[0] < self.max_docs:
            timepoint = self.get_padding(timepoint)
            
        return timepoint, label

    def __len__(self):
        # Returns the total number of timepoints in the data
        n_timepoints = self.n_samples*args.timepoints
    
        return n_timepoints


# Creates test dataset
test_data = ClassificationDataset(ts_sets, max_docs, len(ts_sets), 'test')

# Creates dataloader for test data
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

##############Define the network modules############

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

# Define the RNN network class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_output_size, num_layers, output_size):
        super(RNN, self).__init__()

        # define network layers    
        self.blstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=args.lstm_dropout, bidirectional=True, batch_first=True)  
        self.fc = nn.Linear(lstm_output_size, int(lstm_output_size/2))
        self.fc2 = nn.Linear(int(lstm_output_size/2), output_size)
        
    # Define forward pass
    def forward(self, x):
        # the first value returned by LSTM is all of the hidden states throughout the sequence
        x, _ = self.blstm(x)
        x = x.reshape((x.shape[0],-1))
        x =  x = F.relu(self.fc(x))
        x =  x = F.relu(self.fc2(x))
        
        return x
    
# Define the Classifier network class
class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()

        # define network layers
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    # Define forward pass
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x

# Define input and output size for the fully connected layers
fc_input_size = max_docs*args.emb_size
fc_output_size = args.te_output_size

rnn_batch_size = int(args.batch_size / args.timepoints)

# Define the output size for LSTM
lstm_output_size = int(2 * args.hidden_size*(args.batch_size / rnn_batch_size))

if args.smoothing_type == 'no':
    rnn_input_size = fc_output_size
elif args.smoothing_type == 'double':
    rnn_input_size = fc_output_size * 3
else:
    rnn_input_size = fc_output_size * 2
    
# Instantiate the models
tp_emb = TimepointEmbedding(fc_input_size, fc_output_size, args.emb_size, args.n_heads, args.attn_layers, args.attn_dropout).to(device)
rnn = RNN(rnn_input_size, args.hidden_size, lstm_output_size, args.lstm_layers, args.rnn_output_size).to(device)
cl = Classifier(args.rnn_output_size, args.timepoints).to(device)

# Print model architectures
#print('TimepointEmbedding: ', tp_emb)
#print('RNN: ', rnn)

# Define optimizer and loss function
optimizer = torch.optim.SGD(cl.parameters(), lr=args.lr)

criterion = nn.BCELoss()

# Load model and optimizer
print('Loading model from {}'.format(args.model_path))
checkpoint = torch.load(args.model_path)
tp_emb.load_state_dict(checkpoint['tp_emb_state_dict'])
rnn.load_state_dict(checkpoint['rnn_state_dict'])
cl.load_state_dict(checkpoint['cl_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print('Model loaded succesfully.')

# Function for applying smoothing to the timepoint level data
def smoothen_data(data):
    pad = int((args.window_size - 1) / 2)
    data_padded = F.pad(input=data, pad=(0,0,pad,pad), mode='constant', value=0)
    rolled_data = data_padded.unfold(dimension=0, size=args.window_size, step=1)

    if args.smoothing_type == 'mean':
        tmp = rolled_data.mean(dim=2)
    elif args.smoothing_type == 'std':
        tmp = rolled_data.std(dim=2)
    elif args.smoothing_type == 'double':
        t = rolled_data.mean(dim=2)
        t2 = rolled_data.std(dim=2)
        tmp = torch.cat((t, t2), 1)

    res = torch.cat((data, tmp), 1)

    return res

# Function for testing the model
def test_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    tp_emb.eval()
    rnn.eval()
    
    rand_scores = []
    precs = []
    recalls = []

    num_batches = len(test_dataloader)
    
    skipped = 0
    
    # Gradients are not calculated during testing
    with torch.no_grad():
        
        # Loop over test data batches
        for (x_test, y_test) in test_dataloader:
            
            true_pivots = list(np.where(y_test[:-1] != y_test[1:])[0] + 1)
            
            if len(true_pivots) > 1:
            
                # Calculate output
                tp_emb_output = tp_emb(x_test)
                batch_size = int(tp_emb_output.shape[0] / args.timepoints)
                
                if args.smoothing_type != 'no':
                    tp_emb_output = smoothen_data(tp_emb_output)

                rnn_input = tp_emb_output.reshape(batch_size, -1, tp_emb_output.shape[1])
                rnn_output = rnn(rnn_input)

                cl_output = cl(rnn_output)
                preds = torch.round(cl_output.detach())[0,:]

                pred_pivots = list(np.where(preds[:-1] != preds[1:])[0] + 1)
            
                true_pivots.append(100)
                pred_pivots.append(100)
                
                prec, recall = precision_recall(true_pivots, pred_pivots, margin=args.pr_margin)
                rand = randindex(true_pivots, pred_pivots)

                precs.append(prec)
                recalls.append(recall)
                rand_scores.append(rand)
            
            else:
                skipped += 1
    
    return np.mean(rand_scores), np.mean(precs), np.mean(recalls), num_batches, skipped

rand, prec, rec, num_batches, skipped = test_model()

print("Rand score of supervised pivot point detection: %.3f"%rand)
print("Precision of supervised pivot point detection: %.3f"%prec)
print("Recall of supervised pivot point detection: %.3f"%rec)
print('Number of datasets: ', num_batches - skipped)
print('Skipped: ', skipped)

results = {'rand': rand, 'prec': prec, 'rec': rec, 'args': args}

# Save stats
file_to_write = open(args.res_path + args.job_name + '.pkl', 'wb')
pickle.dump(results, file_to_write)