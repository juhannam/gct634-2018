# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import sys
import os
import numpy as np
import time
import argparse

from model import *
from data_loader import *
from preprocessing import *
from train import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# gpu_option
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_use', type=int, help='GPU enable')
parser.add_argument('--which_gpu', type=int, help='GPU enable')
args = parser.parse_args()
print(args)

# options
melBins = 128
hop = 512
frames = int(29.9*22050.0/hop)
batch_size = 5
learning_rate = 0.01
num_epochs = 50

# A location where labels and features are located
label_path = './gtzan/'
mel_path = './gtzan_mel/'

def main():

    # load normalized mel-spectrograms and labels
    x_train,y_train,x_valid,y_valid,x_test,y_test = load_data(label_path, mel_path, melBins, frames)
    print(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape,x_test.shape,y_test.shape)

    # data loader    
    train_data = gtzandata(x_train,y_train)
    valid_data = gtzandata(x_valid,y_valid)
    test_data = gtzandata(x_test,y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # load model
    if args.gpu_use == 1:
        model = model_1DCNN().cuda(args.which_gpu)
    elif args.gpu_use == 0:
        model = model_1DCNN()
    print(model)

    # loss function 
    criterion = nn.CrossEntropyLoss()

    # run
    start_time = time.time()
    fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs,args)
    print("--- %s seconds spent ---" % (time.time() - start_time))

    # evaluation
    avg_loss, output_all, label_all = eval(model,test_loader,criterion,args)
    prediction = np.concatenate(output_all)
    prediction = prediction.reshape(len(y_test),10)
    prediction = prediction.argmax(axis=1)

    y_label = np.concatenate(label_all)

    comparison = prediction - y_label
    acc = float(len(y_test) - np.count_nonzero(comparison)) / len(y_test)
    print('Test Accuracy: {:.4f} \n'. format(acc))

if __name__ == '__main__':
    main()






