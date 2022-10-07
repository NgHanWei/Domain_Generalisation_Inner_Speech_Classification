import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from cmath import inf
from vae.utils import *
from vae_models import vanilla_vae

import mne 
import pickle
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt

from braindecode.models.deep4 import Deep5Net
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
import torch.nn.functional as F
from os.path import join as pjoin
import argparse
import json
import logging
import sys

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time

mne.set_log_level(verbose='warning') #to avoid info at terminal
warnings.filterwarnings(action = "ignore", category = DeprecationWarning ) 
warnings.filterwarnings(action = "ignore", category = FutureWarning ) 

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
set_random_seeds(seed=2022, cuda=True)

parser = argparse.ArgumentParser(
    description='Subject-adaptative classification with Inner Speech')
parser.add_argument('-subj', type=int,
                    help='Target Subject', required=True)
parser.add_argument('--meta',default=False, help='Training Mode', action='store_true')
args = parser.parse_args()

### Hyperparameters

# The root dir have to point to the folder that cointains the database
root_dir = "D:/Imagined_speech/"

# Data Type
datatype = "EEG"
# Sampling rate
fs = 256
# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5

# Target Subject
targ_subj = args.subj #[1 to 10]
meta = args.meta

X_train = np.array([])
Y_train = np.array([])
X_val = np.array([])
Y_val = np.array([])
for i in range(1,11):

    # Subject number
    N_S = i   #[1 to 10]

    #@title Data extraction and processing

    # Load all trials for a sigle subject
    X, Y = Extract_data_from_subject(root_dir, N_S, datatype)

    # Cut usefull time. i.e action interval
    X = Select_time_window(X = X, t_start = t_start, t_end = t_end, fs = fs)

    # print("Data shape: [trials x channels x samples]")
    # print(X.shape) # Trials, channels, samples

    # print("Labels shape")
    # print(Y.shape) # Time stamp, class , condition, session

    # Conditions to compared
    Conditions = [["Inner"],["Inner"],["Inner"],["Inner"]]
    # The class for the above condition
    Classes    = [  ["Up"] ,["Down"],["Left"],["Right"] ]

    # Transform data and keep only the trials of interes
    X , Y =  Transform_for_classificator(X, Y, Classes, Conditions)

    print("Final data shape")
    print(X.shape)

    print("Final labels shape")
    print(Y.shape)

    # Normalize Data
    Max_val = 500
    norm = np.amax(abs(X))
    X = Max_val * X/norm

    subjs = [1,2,3,4,5,6,7,8,9,10]

    if i == targ_subj:
        X_test = X[100:]
        Y_test = Y[100:]
        X_adapt = X[:100]
        Y_adapt = Y[:100]
    elif i == subjs[targ_subj-2]:
        X_val = X[:].astype(np.float32)
        Y_val = Y[:].astype(np.int64)
    else:
        X_train = np.concatenate((X_train, X),axis=0) if X_train != [] else X
        Y_train = np.concatenate((Y_train, Y),axis=0) if Y_train != [] else Y
        # X_val = np.concatenate((X_val, X),axis=0) if X_val != [] else X[round(len(X)*0.9):]
        # Y_val = np.concatenate((Y_val, Y),axis=0) if Y_val != [] else Y[round(len(Y)*0.9):]


print("Training Data shape")    
print(X_train.shape)
print(Y_train.shape)

print("Test Data shape")
print(X_test.shape)
print(Y_test.shape)

X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.int64)
X_valid = X_val.astype(np.float32)
Y_valid = Y_val.astype(np.int64)
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.int64)
X_adapt = X_adapt.astype(np.float32)
Y_adapt = Y_adapt.astype(np.int64)

X_train = torch.from_numpy(X_train)
X_valid = torch.from_numpy(X_valid)
X_test = torch.from_numpy(X_test)
X_train = X_train[:,np.newaxis,:,:].to('cuda')
X_valid = X_valid[:,np.newaxis,:,:].to('cuda')
X_test = X_test[:,np.newaxis,:,:].to('cuda')

X_adapt = torch.from_numpy(X_adapt)
X_adapt = X_adapt[:,np.newaxis,:,:].to('cuda')

# X_train = []
# X_valid = []
# X_test = []

# VAE model
input_shape=(X_adapt.shape[1:])
batch_size = 16
kernel_size = 5
filters = 8
features = 16
data = 'eeg'
clip = 0

targ_subj = 1

data_load = torch.split(X_adapt,batch_size)
channels = len(X_adapt[0,0,:,0])

print("Number of Features: " + str(features))
if clip > 0:
    print("Gradient Clipping: " + str(clip))
else:
    print("No Gradient Clipping")

if data == 'eeg':
    print("Data Loaded: " + data)

# learning parameters
epochs = 200
lr = 0.0005
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Define Model
model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=data,data_length=len(data_load[0][:,0,0,0])).to(device)
# print(model)
optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.5, 0.999),weight_decay=0.5*lr)
criterion = nn.BCELoss(reduction='sum')

## Number of trainable params
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable params: " + str(pytorch_total_params))

# Save file name
if os.path.exists("./trained_vae/") == False:
    os.makedirs("./trained_vae/")
file_name = "./trained_vae/vae_torch" +  '_' + str(data)  + '_' + str(targ_subj) + '_' + str(filters) + '_' + str(channels) + '_' + str(features) + ".pt"

def recon_loss(outputs,targets):

    outputs = torch.flatten(outputs)
    targets = torch.flatten(targets)

    loss = nn.MSELoss()

    recon_loss = loss(outputs,targets)

    return recon_loss

def recon_loss_2(outputs,targets):

    loss = nn.MSELoss()

    for i in range(0,outputs.shape[0]):
        original = torch.flatten(outputs[i,:,:,:])
        target = torch.flatten(targets[i,:,:,:])
        recon_loss += loss(original,target)

    recon_loss = recon_loss/outputs.shape[0]

    return recon_loss


def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def fit(model):
    model.train()
    running_loss = 0.0
    # For each batch
    for batch in tqdm(range(0,len(data_load))):
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data_load[batch])
        bce_loss = recon_loss(reconstruction,data_load[batch])
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    train_loss = running_loss/len(X_adapt)

    return train_loss

def validate(model):
    model.eval()
    running_loss = 0.0
    full_recon_loss = 0.0
    with torch.no_grad():
        # For each image in batch
        # for batch in range(0,len(X_valid)):
        reconstruction, mu, logvar = model(X_valid)
        bce_loss = recon_loss(reconstruction,X_valid)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        full_recon_loss += bce_loss.item()

    val_loss = running_loss/len(X_valid)
    full_recon_loss = full_recon_loss/len(X_valid)
    print(f"Recon Loss: {full_recon_loss:.4f}")
    return val_loss, full_recon_loss

def eval(model):
    model.eval()
    nll_loss_eval = 0
    with torch.no_grad():
        reconstruction, mu, logvar = model(X_test)
        recon_1 = recon_loss(reconstruction,X_test)
        KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        nll_loss_eval = (recon_1 + KLD_1)/len(X_test)
        return nll_loss_eval.item()

model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=data,data_length=len(data_load[0][:,0,0,0])).to(device)
model.load_state_dict(torch.load(file_name))

model.eval()
with torch.no_grad():
    # print(X_test[:1].shape)
    
    reconstruction, mu, logvar = model(X_test[:10])
    plt.imshow(X_test[1,0,:,:].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(reconstruction[1,0,:,:].detach().cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.show()
    bce_loss = recon_loss(reconstruction,X_test[:10])
    print(bce_loss)

    total_bce = 0
    count = 0
    for batch in tqdm(range(0,len(data_load))):
        count = count + 1
        reconstruction, mu, logvar = model(data_load[batch])
        batch_bce_loss = recon_loss(reconstruction,data_load[batch])
        total_bce += batch_bce_loss

    print(total_bce/count)

    data_load = torch.split(X_train,batch_size)
    total_bce = 0
    count = 0
    for batch in tqdm(range(0,len(data_load))):
        count = count + 1
        reconstruction, mu, logvar = model(data_load[batch])
        batch_bce_loss = recon_loss(reconstruction,data_load[batch])
        total_bce += batch_bce_loss

    print(total_bce/count)