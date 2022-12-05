#!/usr/bin/env python
# coding: utf-8
'''Subject-adaptative classification with KU Data,
using Deep ConvNet model from [1].

References
----------
.. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
   Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
   Deep learning with convolutional neural networks for EEG decoding and
   visualization.
   Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
'''
import argparse
import json
import logging
import sys
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from braindecode.models.deep4 import Deep4Net
from braindecode.models.eegnet import EEGNetv4
from braindecode.torch_ext.optimizers import AdamW
from braindecode.torch_ext.util import set_random_seeds
from torch import nn
from vae_models import vanilla_vae
import torch.optim as optim
from sklearn.utils import shuffle

from Inner_Speech_Dataset.Python_Processing.Data_extractions import  Extract_data_from_subject
from Inner_Speech_Dataset.Python_Processing.Data_processing import  Select_time_window, Transform_for_classificator, Split_trial_in_time

# python train_adapt.py -scheme 5 -trfrate 10 -subj $subj

logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.INFO, stream=sys.stdout)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
set_random_seeds(seed=2022, cuda=True)

parser = argparse.ArgumentParser(
    description='Subject-adaptative classification with Inner Speech')
parser.add_argument('--eegnet',default=False, help='Training Model', action='store_true')
parser.add_argument('-scheme', type=int, help='Adaptation scheme', default=4)
parser.add_argument(
    '-trfrate', type=int, help='The percentage of data for adaptation', default=100)
parser.add_argument('--dgtrain',default=False, help='Domain Generalisation on Training', action='store_true')
parser.add_argument('--dgval',default=False, help='Domain Generalisation on Validation', action='store_true')
parser.add_argument('--dgtest',default=False, help='Domain Generalisation on Test', action='store_true')
parser.add_argument('-lr', type=float, help='Learning rate', default=0.0005)
parser.add_argument('-gpu', type=int, help='The gpu device to use', default=0)

args = parser.parse_args()
outpath = './adapt_results/'
modelpath = './EEGNET/coeff1_dgtrain/'
scheme = args.scheme
rate = args.trfrate
lr = args.lr
dgtrain = args.dgtrain
dgval = args.dgval
dgtest = args.dgtest
eegnet = args.eegnet
torch.cuda.set_device(args.gpu)
set_random_seeds(seed=2022, cuda=True)
BATCH_SIZE = 16
TRAIN_EPOCH = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
kernel_size = 5
filters = 8
features = 16
data = 'eeg'

root_dir = "D:/Imagined_speech/"

# Data Type
datatype = "EEG"

# Sampling rate
fs = 256

# Select the useful par of each trial. Time in seconds
t_start = 1.5
t_end = 3.5

all_loss = 0
for subj in range(1,11):

    # Target Subject
    targ_subj = subj #[1 to 10]

    X_train = np.array([])
    Y_train = np.array([])
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

        if i == targ_subj:
            print(X.shape)
            # p = np.random.permutation(len(Y))
            # X = X[p]
            # Y = Y[p]
            X, Y = shuffle(X, Y, random_state=0)
            X_test = X
            Y_test = Y
        else:
            X_train = np.concatenate((X_train, X),axis=0) if len(X_train) > 0 else X
            Y_train = np.concatenate((Y_train, Y),axis=0) if len(Y_train) > 0 else Y

    print("Training Data shape")    
    print(X_train.shape)
    print(Y_train.shape)

    print("Test Data shape")
    print(X_test.shape)
    print(Y_test.shape)

    X_train2 = X_train.astype(np.float32)
    Y_train2 = Y_train.astype(np.int64)
    X_test = X_test.astype(np.float32)
    Y_test = Y_test.astype(np.int64)

    channels = len(X_train[0,:,0])
    file_name = "./trained_vae/vae_torch" +  '_' + str(data)  + '_' + str(targ_subj) + '_' + str(filters) + '_' + str(channels) + '_' + str(features) + ".pt"

    n_classes = 4
    in_chans = X.shape[1]
    # final_conv_length = auto ensures we only get a single output in the time dimension
    if eegnet == True:
        model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=X.shape[2],
                        final_conv_length='auto').cuda()
    else:
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                        input_time_length=X.shape[2],
                        final_conv_length='auto').cuda()

    def reset_conv_pool_block(network, block_nr):
        suffix = "_{:d}".format(block_nr)
        conv = getattr(network, 'conv' + suffix)
        kernel_size = conv.kernel_size
        n_filters_before = conv.in_channels
        n_filters = conv.out_channels
        setattr(network, 'conv' + suffix,
                nn.Conv2d(
                    n_filters_before,
                    n_filters,
                    kernel_size,
                    stride=(1, 1),
                    bias=False,
                ))
        setattr(network, 'bnorm' + suffix,
                nn.BatchNorm2d(
                    n_filters,
                    momentum=0.1,
                    affine=True,
                    eps=1e-5,
                ))
        # Initialize the layers.
        conv = getattr(network, 'conv' + suffix)
        bnorm = getattr(network, 'bnorm' + suffix)
        nn.init.xavier_uniform_(conv.weight, gain=1)
        nn.init.constant_(bnorm.weight, 1)
        nn.init.constant_(bnorm.bias, 0)


    def reset_model(checkpoint):
        # Load the state dict of the model.
        model.network.load_state_dict(checkpoint['model_state_dict'])

        if scheme != 5:
            # Freeze all layers.
            for param in model.network.parameters():
                param.requires_grad = False

            if scheme in {1, 2, 3, 4}:
                # Unfreeze the FC layer.
                for param in model.network.conv_classifier.parameters():
                    param.requires_grad = True

            if scheme in {2, 3, 4}:
                # Unfreeze the conv4 layer.
                for param in model.network.conv_4.parameters():
                    param.requires_grad = True
                for param in model.network.bnorm_4.parameters():
                    param.requires_grad = True

            if scheme in {3, 4}:
                # Unfreeze the conv3 layer.
                for param in model.network.conv_3.parameters():
                    param.requires_grad = True
                for param in model.network.bnorm_3.parameters():
                    param.requires_grad = True

            if scheme == 4:
                # Unfreeze the conv2 layer.
                for param in model.network.conv_2.parameters():
                    param.requires_grad = True
                for param in model.network.bnorm_2.parameters():
                    param.requires_grad = True

        # Only optimize parameters that requires gradient.
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                        lr=lr, weight_decay=0.5*0.001)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.compile(loss=F.nll_loss, optimizer=optimizer,
                    iterator_seed=2022, )

    cutoff = int(rate * 50 / 100)
    # Use only session 1 data for training
    assert(cutoff <= 50)

    total_loss = []

    checkpoint = torch.load(pjoin(modelpath, 'DG_model_subj' + str(targ_subj) + '.pt'),
                            map_location='cuda:' + str(args.gpu))
    if eegnet == False:
        reset_model(checkpoint)
    else:
        model.network.load_state_dict(checkpoint['model_state_dict'])
        # Only optimize parameters that requires gradient.
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.network.parameters()),
                        lr=lr, weight_decay=0.5*0.001)
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.compile(loss=F.nll_loss, optimizer=optimizer,
                    iterator_seed=2022, )

    X, Y = X_test, Y_test
    X_train, Y_train = X[:cutoff], Y[:cutoff]

    model_vae = vanilla_vae.VanillaVAE(filters=filters,channels=channels,features=features,data_type=data,data_length=len(X_train[:,0,0])).to(device)
    model_vae.load_state_dict(torch.load(file_name))
    optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.5, 0.999),weight_decay=0.5*lr)

    X_new = torch.from_numpy(X_train)
    X_new = X_new[:,np.newaxis,:,:].to('cuda')
    data_load = torch.split(X_new,batch_size)

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
        for batch in range(0,len(data_load)):
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(data_load[batch])
            bce_loss = recon_loss(reconstruction,data_load[batch])
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = running_loss/len(X_train)

        return train_loss

    # epochs = 20
    # for epoch in range(epochs):
    #     print(f"Epoch {epoch+1} of {epochs}")
    #     train_epoch_loss = fit(model_vae)

    def domain_adapt(x_array):
        new_X = x_array
        Max_val = 500
        norm = np.amax(abs(new_X))
        new_X = Max_val * new_X/norm
        new_X = torch.from_numpy(new_X)
        new_X = new_X[:,np.newaxis,:,:].to('cuda')
        model_vae.eval()
        with torch.no_grad():
            reconstruction, mu, logvar = model_vae(new_X)
        new_X = reconstruction.detach().cpu().numpy()
        new_X= new_X[:,0,:,:]

        return new_X

    if dgtrain == True:   
        X_train = domain_adapt(X_train)

    # X_train = np.concatenate((X_train, X_train_new),axis=0)
    # Y_train = np.concatenate((Y_train, Y_train),axis=0)

    X_val, Y_val = X[cutoff:], Y[cutoff:]

    if dgval == True:
        X_val = domain_adapt(X_val)

    model.fit(X_train, Y_train, epochs=TRAIN_EPOCH,
                batch_size=BATCH_SIZE, scheduler='cosine',
                validation_data=(X_val, Y_val), remember_best_column='valid_misclass')
    model.epochs_df.to_csv(pjoin(outpath, 'epochs' + str(targ_subj) + '.csv'))

    if dgtest == True:
        X_test = domain_adapt(X_test)

    test_loss = model.evaluate(X_test[cutoff:], Y_test[cutoff:])
    total_loss.append(test_loss["misclass"])
    with open(pjoin(outpath, 'test' + str(targ_subj) + '.json'), 'w') as f:
        json.dump(test_loss, f)

    print(total_loss[0])
    all_loss += total_loss[0]

print(all_loss/10)