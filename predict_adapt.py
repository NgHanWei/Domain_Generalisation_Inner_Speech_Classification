from cgi import test
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
from braindecode.models.eegnet import EEGNetv4
from braindecode.datautil.signal_target import SignalAndTarget
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
import torch.nn.functional as F
from os.path import join as pjoin
import argparse
import json
import logging
import sys

from sklearn.utils import shuffle

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
parser.add_argument('-coeff', type=float,
                    help='Data Selection Threshold Multiplier, set to 0 for no Selection', default=1)
parser.add_argument('--eegnet',default=False, help='Training Model', action='store_true')
parser.add_argument('--dgtrain',default=False, help='Domain Generalisation on Training', action='store_true')
parser.add_argument('--dgval',default=False, help='Domain Generalisation on Validation', action='store_true')
parser.add_argument('--dgtest',default=False, help='Domain Generalisation on Test', action='store_true')
args = parser.parse_args()

### Hyperparameters

# The root dir have to point to the folder that cointains the database
root_dir = "./"

# Data Type
datatype = "EEG"
# Sampling rate
fs = 256
# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5

# Target Subject
targ_subj = args.subj #[1 to 10]
eegnet = args.eegnet
dgtrain = args.dgtrain
dgval = args.dgval
dgtest = args.dgtest
coeff = args.coeff

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
        X, Y = shuffle(X, Y, random_state=0)
        X_test = X[:]
        Y_test = Y[:]
        X_adapt = X[:50]
        Y_adapt = Y[:50]
    elif i == subjs[targ_subj-2]:
        X_val = X[:].astype(np.float32)
        Y_val = Y[:].astype(np.int64)
    else:
        X_train = np.concatenate((X_train, X),axis=0) if X_train != [] else X
        Y_train = np.concatenate((Y_train, Y),axis=0) if Y_train != [] else Y


print("Training Data shape")    

original_trials = X_train.shape[0]
original_X_train = X_train

print("Test Data shape")
print(X_test.shape)
print(Y_test.shape)
print(np.unique(Y_test))

X_train = X_train.astype(np.float32)
Y_train = Y_train.astype(np.int64)
X_valid = X_val.astype(np.float32)
Y_valid = Y_val.astype(np.int64)
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.int64)
X_adapt = X_adapt.astype(np.float32)
Y_adapt = Y_adapt.astype(np.int64)

print(np.median(X_train))

X_train = torch.from_numpy(X_train)
X_valid = torch.from_numpy(X_valid)
X_test = torch.from_numpy(X_test)
X_train = X_train[:,np.newaxis,:,:].to('cuda')
X_valid = X_valid[:,np.newaxis,:,:].to('cuda')
X_test = X_test[:,np.newaxis,:,:].to('cuda')

X_adapt = torch.from_numpy(X_adapt)
X_adapt = X_adapt[:,np.newaxis,:,:].to('cuda')
print(X_train.shape)
print(Y_train.shape)

# VAE model
input_shape=(X_adapt.shape[1:])
batch_size = 16
kernel_size = 5
filters = 8
features = 16
data = 'eeg'
clip = 0

data_load = torch.split(X_train,batch_size)
data_load_check = torch.split(X_train,1)
channels = len(X_train[0,0,:,0])

print("Number of Features: " + str(features))
if clip > 0:
    print("Gradient Clipping: " + str(clip))
else:
    print("No Gradient Clipping")

if data == 'eeg':
    print("Data Loaded: " + data)

# learning parameters
epochs = 20
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
        # print(data_load[batch].shape)
        reconstruction, mu, logvar = model(data_load[batch])
        bce_loss = recon_loss(reconstruction,data_load[batch])
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    train_loss = running_loss/len(X_train)

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
        reconstruction, mu, logvar = model(X_train)
        recon_1 = recon_loss(reconstruction,X_train)
        KLD_1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        nll_loss_eval = (recon_1 + KLD_1)/len(X_train)
        return nll_loss_eval.item()

# Train the Model
train_loss = []
val_loss = []
eval_loss = []
best_val_loss = inf
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model)
    val_epoch_loss, full_recon_loss = validate(model)
    # eval_epoch_loss = eval(model)
    
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    # eval_loss.append(eval_epoch_loss)

    #Save best model
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        torch.save(model.state_dict(),file_name)
        print(f"Saving Model... Best Val Loss: {best_val_loss:.4f}")

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}")

model = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=data,data_length=len(data_load[0][:,0,0,0])).to(device)
model.load_state_dict(torch.load(file_name))

data_load = torch.split(X_test,len(X_test))
model.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model(data_load[0])
mu = mu.cpu().detach().numpy()
print(mu.shape)
# plot_clustering(mu, Y_test, engine='matplotlib', download = False)

model.eval()
with torch.no_grad():
    # print(X_test[:1].shape)
    reconstruction, mu, logvar = model(X_test[:50])
    test_bce_loss = recon_loss(reconstruction,X_test[:50])
    print(test_bce_loss)

    total_bce = 0
    count = 0
    X_train_new = []
    Y_train_new = []
    anomaly_list = []
    for batch in tqdm(range(0,len(data_load_check))):    
        # print(data_load_check[batch].shape)
        reconstruction, mu, logvar = model(data_load_check[batch])
        # X_train_new = np.concatenate((X_train_new, reconstruction.detach().cpu().numpy()),axis=0) if X_train_new != [] else reconstruction.detach().cpu().numpy()
        batch_bce_loss = recon_loss(reconstruction,data_load_check[batch])
        # print(batch_bce_loss)
        total_bce += batch_bce_loss

        add = 0
        if coeff == 0:
            add = inf

        if batch_bce_loss < test_bce_loss*coeff + add:
            print(batch_bce_loss)
            anomaly_list.append(count)
            if dgtrain == True:
                X_train_new = np.concatenate((X_train_new, reconstruction.detach().cpu().numpy()),axis=0) if X_train_new != [] else reconstruction.detach().cpu().numpy()
                Y_train_new = np.concatenate((Y_train_new, [Y_train[count]]),axis=0) if Y_train_new != [] else [Y_train[count]]

            else:
                X_train_new = np.concatenate((X_train_new, data_load_check[batch].detach().cpu().numpy()),axis=0) if X_train_new != [] else data_load_check[batch].detach().cpu().numpy()
                Y_train_new = np.concatenate((Y_train_new, [Y_train[count]]),axis=0) if Y_train_new != [] else [Y_train[count]]

        count = count + 1

    if X_train_new == []:
        X_train_new = reconstruction.detach().cpu().numpy()
        Y_train_new = np.asarray([Y_train[count-1]])

    X_train_new = X_train_new[:,0,:,:]

    print(X_train_new.shape)
    print(Y_train_new.shape)

    print(total_bce/count)

    data_load = torch.split(X_adapt,batch_size)
    total_bce = 0
    count = 0
    for batch in tqdm(range(0,len(data_load))):
        count = count + 1
        reconstruction, mu, logvar = model(data_load[batch])
        batch_bce_loss = recon_loss(reconstruction,data_load[batch])
        total_bce += batch_bce_loss

    print(total_bce/count)



### Hyperparameters

# Data Type
datatype = "EEG"
# Sampling rate
fs = 256
# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5

# Target Subject
targ_subj = args.subj #[1 to 10]
meta = False

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
        X, Y = shuffle(X, Y, random_state=0)
        X_test = X[50:]
        Y_test = Y[50:]
    elif i == subjs[targ_subj-2]:
        X_val = X[:].astype(np.float32)
        Y_val = Y[:].astype(np.int64)
    else:
        X_train = np.concatenate((X_train, X),axis=0) if X_train != [] else X
        Y_train = np.concatenate((Y_train, Y),axis=0) if Y_train != [] else Y

print("Training Data shape")    
print(X_train.shape)
print(Y_train.shape)

print("Test Data shape")
print(X_test.shape)
print(Y_test.shape)

Max_val = 500
norm = np.amax(abs(X_train_new))
X_train_new = Max_val * X_train_new/norm
X_train = X_train_new.astype(np.float32)
Y_train = Y_train_new.astype(np.int64)

print(Y_train.shape)

X_val = X_val.astype(np.float32)
Y_val = Y_val.astype(np.int64)
X_test = X_test.astype(np.float32)
Y_test = Y_test.astype(np.int64)


# Domain adaptation on valdiation data
model_vae = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=data,data_length=len(data_load[0][:,0,0,0])).to(device)
model_vae.load_state_dict(torch.load(file_name))

new_X = X_val
Max_val = 500
norm = np.amax(abs(new_X))
new_X = Max_val * new_X/norm
new_X = torch.from_numpy(new_X)
new_X = new_X[:,np.newaxis,:,:].to('cuda')
model_vae.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model_vae(new_X)
new_X = reconstruction.detach().cpu().numpy()
X_val_new= new_X[:,0,:,:]

### Training Details
TRAIN_EPOCH = 50
BATCH_SIZE = 16

# print(X_train)
# print(X_val)
# print(X_test)

train_set = SignalAndTarget(X_train, y=Y_train)
if dgval == True:
    valid_set = SignalAndTarget(X_val_new, y=Y_val)
else:
    valid_set = SignalAndTarget(X_val, y=Y_val)

new_X = X_test
Max_val = 500
norm = np.amax(abs(new_X))
new_X = Max_val * new_X/norm
new_X = torch.from_numpy(new_X)
new_X = new_X[:,np.newaxis,:,:].to('cuda')
model_vae.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model_vae(new_X)
new_X = reconstruction.detach().cpu().numpy()
X_test_new= new_X[:,0,:,:]

if dgtest == True:
    test_set = SignalAndTarget(X_test_new, y=Y_test)
else:
    test_set = SignalAndTarget(X_test, y=Y_test)
n_classes = 4
in_chans = train_set.X.shape[1]

if eegnet == True:
    model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,
                    input_time_length=train_set.X.shape[2],
                    final_conv_length='auto').cuda()
else:

    model = Deep5Net(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=train_set.X.shape[2],
                        final_conv_length='auto').cuda()

optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001)
model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )


exp = model.fit(train_set.X, train_set.y, epochs=TRAIN_EPOCH, batch_size=BATCH_SIZE, scheduler='cosine',
                    validation_data=(valid_set.X, valid_set.y), remember_best_column='valid_loss', meta=meta)
rememberer = exp.rememberer
base_model_param = {
    'epoch': rememberer.best_epoch,
    'model_state_dict': rememberer.model_state_dict,
    'optimizer_state_dict': rememberer.optimizer_state_dict,
    'loss': rememberer.lowest_val
}
torch.save(base_model_param, pjoin(
    './results/', 'DG_model_subj{}.pt'.format(targ_subj)))
model.epochs_df.to_csv(
    pjoin('./results/', 'DG_epochs_subj{}.csv'.format(targ_subj)))

new_X = test_set.X
Max_val = 500
norm = np.amax(abs(new_X))
new_X = Max_val * new_X/norm
new_X = torch.from_numpy(new_X)
model_vae.eval()
with torch.no_grad():
    reconstruction, mu, logvar = model_vae(new_X[:,np.newaxis,:,:].to('cuda'))
new_X = reconstruction.detach().cpu().numpy()
print(new_X.shape)

# Comment out to not apply recon on test
if dgtest == True:
    test_set = SignalAndTarget(new_X[:,0,:,:], y=Y_test)

test_loss = model.evaluate(test_set.X, test_set.y)
print(test_loss)
with open(pjoin('./results/', 'DG_test_base_subj{}.json'.format(targ_subj)), 'w') as f:
    json.dump(test_loss, f)