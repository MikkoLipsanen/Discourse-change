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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.cluster import adjusted_rand_score

# Uses GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

parser = argparse.ArgumentParser(description='Supervised timepoint classification')

parser.add_argument('--data_path', type=str, default='data/samples/', help='file containing data')
parser.add_argument('--res_path', type=str, default='results/', help='path to save results')
parser.add_argument('--model_path', type=str, default='model/contrastive_model.pt', help='path to save the model')
parser.add_argument('--save_model_path', type=str, default='model/supervised_tp_model.pt', help='path to save the model')
parser.add_argument('--batch_size', type=int, default=3000, help='defines batch size')
parser.add_argument('--timepoints', type=int, default=100, help='defines number of timepoints in sample set')
parser.add_argument('--tr_size', type=int, default=3000, help='defines the size of the training dataset')
parser.add_argument('--val_size', type=int, default=1000, help='defines the size of the validation dataset')
parser.add_argument('--ts_size', type=int, default=1000, help='defines the size of the test dataset')
parser.add_argument('--lr', type=float, default=0.1, help='defines the learning rate')
parser.add_argument('--epochs', type=int, default=20, help='defines the number of epochs')
parser.add_argument('--hidden_size', type=int, default=10, help='bLSTM hidden state size')
parser.add_argument('--lstm_layers', type=int, default=2, help='number of bLSTM layers')
parser.add_argument('--emb_size', type=int, default=300, help='Size of the document embedding vector')
parser.add_argument('--te_output_size', type=int, default=300, help='Size of the TimepointEmbedding output vector')
parser.add_argument('--rnn_output_size', type=int, default=300, help='Size of the TimepointEmbedding output vector')
parser.add_argument('--lstm_dropout', type=float, default=0.2, help='dropout probability for bLSTM')
parser.add_argument('--n_heads', type=int, default=2, help='number of attention heads')
parser.add_argument('--attn_layers', type=int, default=4, help='number of TransformerEncoder layers')
parser.add_argument('--attn_dropout', type=float, default=0.2, help='dropout probability for TransformerEncoder')
parser.add_argument('--pos_w', type=float, default=3, help='weight for the loss function')
parser.add_argument('--job_name', type=str, default='supervised_tp_classification', help='name used in the saved result files')
parser.add_argument('--smoothing_type', type=str, default='no', help='Type of smoothing calculation')
parser.add_argument('--window_size', type=int, default=9, help='Size of the smoothing window')

args = parser.parse_args()
print(args)

############Create train, validation and test splits#########

# Load the number indicating max amount of docs per timepoint
a_file = open(args.data_path + "tracker_dict", "rb")
tracker_dict = pickle.load(a_file)
a_file.close()
    
max_docs = tracker_dict['max_docs']
print('Max docs: ', max_docs)

# List all files in train and test folders
tr_val = [f for f in listdir(args.data_path + "train") if isfile(join(args.data_path + "train", f))]
test = [f for f in listdir(args.data_path + "test") if isfile(join(args.data_path + "test", f))]

# Shuffle the order of files in the list
random.shuffle(tr_val)
random.shuffle(test)

# Select defined number of train, validation and test files
tr_sets = tr_val[:args.tr_size]
val_sets = tr_val[args.tr_size:args.tr_size+args.val_size]
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

# Creates train, validation and test datasets
train_data = ClassificationDataset(tr_sets, max_docs, len(tr_sets), 'train')
validation_data = ClassificationDataset(val_sets, max_docs, len(val_sets), 'train')
test_data = ClassificationDataset(ts_sets, max_docs, len(ts_sets), 'test')

# Creates dataloaders for train, validation and test data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
validation_dataloader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Print the batch shape
train_features, train_labels = next(iter(train_dataloader))
print(f"Train batch shape: {train_features.size()}")


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
        x = self.fc2(x)

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

pw = torch.tensor([args.pos_w])
criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

# Keeps count of epochs when training is continued with saved model
epoch_count = 0

## define model and optimizer
print('Loading model from {}'.format(args.model_path))
checkpoint = torch.load(args.model_path)
tp_emb.load_state_dict(checkpoint['tp_emb_state_dict'])
rnn.load_state_dict(checkpoint['rnn_state_dict'])
if 'cl_state_dict' in checkpoint:
    cl.load_state_dict(checkpoint['cl_state_dict'])
    epoch = checkpoint['epoch']
    epoch_count += epoch
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


def eval_results(cl_output, y_train):
    # Calculate the number of correct predictions in batch  
    cl_output = torch.sigmoid(cl_output)
    preds = torch.round(cl_output.detach())
    correct = (torch.sum(preds == y_train.detach())).item()
    prec_seq, rec_seq, f_seq, s_seq = precision_recall_fscore_support(y_train.detach().numpy(), preds.numpy(), average='binary', zero_division=0)
    
    rand = adjusted_rand_score(y_train.detach().numpy(), preds.numpy())
    
    return correct, prec_seq, rec_seq, f_seq, rand
 
#######Define the train, validation and test iteration functions#######

# Function for training the model
def train_model():
    
    # Set the networks to train mode
    tp_emb.train()
    rnn.train()
    
    loss = 0
    correct = 0

    precision = 0
    recall = 0
    f_score = 0
    rand_score = 0
 
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    
    # Loop over train data batches
    for (x_train, y_train) in train_dataloader:

        # Calculate output
        tp_emb_output = tp_emb(x_train)
        batch_size = int(tp_emb_output.shape[0] / args.timepoints)

        if args.smoothing_type != 'no':
            tp_emb_output = smoothen_data(tp_emb_output)

        rnn_input = tp_emb_output.reshape(batch_size, -1, tp_emb_output.shape[1])
        rnn_output = rnn(rnn_input)
        cl_output = cl(rnn_output)
        cl_output = torch.flatten(cl_output)
        
        # Calculate training loss
        tr_loss = criterion(cl_output, y_train)
        loss += tr_loss.item()

        # Reset batch labels between batches
        train_data.reset_batch_labels()

        # Backpropagate loss
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()

        # Calculate the number of correct predictions in batch  
        corr, prec, rec, f, rand = eval_results(cl_output, y_train)
        
        correct += corr
        precision += prec
        recall += rec
        f_score += f
        rand_score += rand

    # Calculate average loss for the epoch
    train_loss = loss / num_batches
    train_accuracy = correct / size

    mean_prec = precision / num_batches
    mean_rec = recall / num_batches
    mean_f = f_score / num_batches
    mean_rand = rand_score / num_batches

    return train_loss, train_accuracy, mean_prec, mean_rec, mean_f, mean_rand

# Function for evaluating the model
def eval_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    tp_emb.eval()
    rnn.eval()
    
    loss = 0
    correct = 0

    precision = 0
    recall = 0
    f_score = 0
    rand_score = 0

    size = len(validation_dataloader.dataset)
    num_batches = len(validation_dataloader)
    
    # Gradients are not calculated during evaluation
    with torch.no_grad():
        
        # Loop over validation data batches
        for (x_val, y_val) in validation_dataloader:
            
            # Calculate output
            tp_emb_output = tp_emb(x_val)
            batch_size = int(tp_emb_output.shape[0] / args.timepoints)

            if args.smoothing_type != 'no':
                tp_emb_output = smoothen_data(tp_emb_output)

            rnn_input = tp_emb_output.reshape(batch_size, -1, tp_emb_output.shape[1])
            rnn_output = rnn(rnn_input)
            cl_output = cl(rnn_output)
            cl_output = torch.flatten(cl_output)
        
            # Calculate loss for test data
            val_loss = criterion(cl_output, y_val)
            loss += val_loss.item()
            
            # Calculate the number of correct predictions in batch  
            corr, prec, rec, f, rand = eval_results(cl_output, y_val)
        
            correct += corr
            precision += prec
            recall += rec
            f_score += f
            rand_score += rand

        # Calculate average loss and accuracy for the epoch
        validation_loss = loss / num_batches
        validation_accuracy = correct / size

        mean_prec = precision / num_batches
        mean_rec = recall / num_batches
        mean_f = f_score / num_batches
        mean_rand = rand_score / num_batches

    return validation_loss, validation_accuracy, mean_prec, mean_rec, mean_f, mean_rand

# Function for testing the model
def test_model():
    
    # Set the networks to evaluation mode (dropout is not applied)
    tp_emb.eval()
    rnn.eval()
    
    loss = 0
    correct = 0
    
    precision = 0
    recall = 0
    f_score = 0
    rand_score = 0

    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    
    # Gradients are not calculated during testing
    with torch.no_grad():
        
        # Loop over test data batches
        for (x_test, y_test) in test_dataloader:
            
            # Calculate output
            tp_emb_output = tp_emb(x_test)
            batch_size = int(tp_emb_output.shape[0] / args.timepoints)

            if args.smoothing_type != 'no':
                tp_emb_output = smoothen_data(tp_emb_output)

            rnn_input = tp_emb_output.reshape(batch_size, -1, tp_emb_output.shape[1])
            rnn_output = rnn(rnn_input)
            cl_output = cl(rnn_output)
            cl_output = torch.flatten(cl_output)
        
            # Calculate loss for test data
            test_loss = criterion(cl_output, y_test)
            loss += test_loss.item()
            
            # Calculate accuracy (portion of labels predicted correctly)
            corr, prec, rec, f, rand = eval_results(cl_output, y_test)
        
            correct += corr
            precision += prec
            recall += rec
            f_score += f
            rand_score += rand

        # Calculate average loss and accuracy for the epoch
        test_loss = loss / num_batches
        test_accuracy = correct / size

        mean_prec = precision / num_batches
        mean_rec = recall / num_batches
        mean_f = f_score / num_batches
        mean_rand = rand_score / num_batches
        
    return test_loss, test_accuracy, mean_prec, mean_rec, mean_f, mean_rand


#########Train the model##########

# Save loss and accuracy for each epoch
tr_losses = []
tr_accuracies = []
val_losses = []
val_accuracies = []

tr_prec = []
tr_rec = []
tr_f = []
tr_rands = []

val_prec = []
val_rec = []
val_f = []
val_rands = []

start_time = time.time()

# Calculate training and evaluation loss and accuracy for each epoch
for epoch in range(epoch_count+1, args.epochs+1): 
    tr_loss, tr_accuracy, tr_mean_prec, tr_mean_rec, tr_mean_f, tr_rand = train_model()
    val_loss, val_accuracy, val_mean_prec, val_mean_rec, val_mean_f, val_rand = eval_model()

    tr_losses.append(tr_loss)
    tr_accuracies.append(tr_accuracy)
    tr_prec.append(tr_mean_prec)
    tr_rec.append(tr_mean_rec)
    tr_f.append(tr_mean_f)
    tr_rands.append(tr_rand)

    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_prec.append(val_mean_prec)
    val_rec.append(val_mean_rec)
    val_f.append(val_mean_f)
    val_rands.append(val_rand)

    # Reset the dataset parameters so that all samples are available for the next epoch
    train_data.reset()
    validation_data.reset()

    epoch_res = "|Epoch %d | Training loss : %.3f | Training accuracy %.3f | Validation loss %.3f | Validation accuracy %.3f"%(epoch, tr_loss, tr_accuracy, val_loss, val_accuracy)
    epoch_r = "|Epoch %d | Training precision : %.3f | Training recall : %.3f | Validation precision %.3f | Validation recall %.3f"%(epoch, tr_mean_prec, tr_mean_rec, val_mean_prec, val_mean_rec)
    epoch_f = "|Epoch %d | Training F-score %.3f | Validation F-score %.3f| Training rand score %.3f | Validation rand score %.3f"%(epoch, tr_mean_f, val_mean_f, tr_rand, val_rand)
	
    # Save results of each epoch to a .txt file
    with open(args.res_path + args.job_name + '_supervised_epoch%i.txt' % epoch, 'w') as f:
        f.write(epoch_res)
        f.write(epoch_r)
        f.write(epoch_f)

    # Save the model checkpoint
    torch.save({
            'epoch': epoch,
            'tr_loss': tr_loss,
            'args': args,
            'tp_emb_state_dict': tp_emb.state_dict(),
            'rnn_state_dict': rnn.state_dict(),
	    'cl_state_dict': cl.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, args.save_model_path)

    print(epoch_res)

print("Mean training accuracy: %.3f" %np.mean(tr_accuracies))

print("Mean validation accuracy: %.3f" %np.mean(val_accuracies))

end_time = time.time()

print("Minutes used for training: ", int((end_time - start_time)/60))

#tp_emb_params = sum(p.numel() for p in tp_emb.parameters() if p.requires_grad)
#rnn_params = sum(p.numel() for p in rnn.parameters() if p.requires_grad)
#print("Number of trainable parameters in TimepointEmbedding model: ", tp_emb_params)
#print("Number of trainable parameters in RNN model: ", rnn_params)

##########Get the model accuracy for the test set#########

test_loss, test_accuracy, test_mean_prec, test_mean_rec, test_mean_f, test_rand = test_model()

print("Test loss : %.3f | Test accuracy %.3f | Test seq F-score %.3f | Test rand score %.3f" %(test_loss, test_accuracy, test_mean_f, test_rand))


##########Save the results#################

results = {'tr_loss': tr_losses, 'tr_accuracy': tr_accuracies, 'val_loss': val_losses, 
           'val_accuracy': val_accuracies, 'tr_prec': tr_prec, 'tr_rec': tr_rec,
           'tr_f': tr_f, 'val_prec': val_prec, 'val_rec': val_rec, 
           'val_f': val_f, 'tr_rand': tr_rands, 'val_rand': val_rands, 'test_loss': test_loss, 'test_accuracy': test_accuracy,
           'test_mean_prec':test_mean_prec, 'test_mean_rec':test_mean_rec, 
           'test_mean_f':test_mean_f, 'test_rand': test_rand, 'args': args}

# Save loss and accuracy
file_to_write = open(args.res_path + 'res_dicts/' + args.job_name + '.pkl', 'wb')
pickle.dump(results, file_to_write)