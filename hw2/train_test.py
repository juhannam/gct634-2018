# GCT634 (2018) HW2
#
# Apr-17-2018: gpu code added version
# 
# Jongpil Lee
#

from __future__ import print_function
import sys
import os
import numpy as np
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# gpu_option
gpu_use = 0
which_gpu = 2

# options
melBins = 128
hop = 512
frames = int(29.9*22050.0/hop)
batch_size = 5
learning_rate = 0.01
num_epochs = 50
num_frames = 120

# A location where labels and features are located
label_path = '/home1/irteam/users/jongpil/data/GTZAN_split/'

# read train / valid / test lists
y_train_dict = {}
y_valid_dict = {}
y_test_dict = {}
with open(label_path + 'train_filtered.txt') as f:
    train_list = f.read().splitlines()
    for line in train_list:
        y_train_dict[line] = line.split('/')[0]
with open(label_path + 'valid_filtered.txt') as f:
    valid_list = f.read().splitlines()
    for line in valid_list:
        y_valid_dict[line] = line.split('/')[0]
with open(label_path + 'test_filtered.txt') as f:
    test_list = f.read().splitlines()
    for line in test_list:
        y_test_dict[line] = line.split('/')[0]


# labels
genres = list(set(y_train_dict.values()+y_valid_dict.values()+y_test_dict.values()))
print(genres)
for iter in range(len(y_train_dict)):
    for iter2 in range(len(genres)):
        if genres[iter2] == y_train_dict[train_list[iter]]:
            y_train_dict[train_list[iter]] = iter2
for iter in range(len(y_valid_dict)):
    for iter2 in range(len(genres)):
        if genres[iter2] == y_valid_dict[valid_list[iter]]:
            y_valid_dict[valid_list[iter]] = iter2
for iter in range(len(y_test_dict)):
    for iter2 in range(len(genres)):
        if genres[iter2] == y_test_dict[test_list[iter]]:
            y_test_dict[test_list[iter]] = iter2


mel_path = '/home1/irteam/users/jongpil/gct634-2018/hw2/gtzan_mel/'

# load data
x_train = np.zeros((len(train_list),melBins,frames))
y_train = np.zeros((len(train_list),))
for iter in range(len(train_list)):
    x_train[iter] = np.load(mel_path + train_list[iter].replace('.wav','.npy'))
    y_train[iter] = y_train_dict[train_list[iter]]

x_valid = np.zeros((len(valid_list),melBins,frames))
y_valid = np.zeros((len(valid_list),))
for iter in range(len(valid_list)):
    x_valid[iter] = np.load(mel_path + valid_list[iter].replace('.wav','.npy'))
    y_valid[iter] = y_valid_dict[valid_list[iter]]

x_test = np.zeros((len(test_list),melBins,frames))
y_test = np.zeros((len(test_list),))
for iter in range(len(test_list)):
    x_test[iter] = np.load(mel_path + test_list[iter].replace('.wav','.npy'))
    y_test[iter] = y_test_dict[test_list[iter]]



# normalize the mel spectrograms
mean = np.mean(x_train)
std = np.std(x_train)
x_train -= mean
x_train /= std
x_valid -= mean
x_valid /= std
x_test -= mean
x_test /= std

# data loader
class gtzandata(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __getitem__(self,index):

        # todo : random cropping audio to 3-second
        #start = random.randint(0,self.x[index].shape[1] - num_frames)
        #mel = self.x[index][:,start:start+num_frames]
        mel = self.x[index]

        entry = {'mel': mel, 'label': self.y[index]}
        
        return entry

    def __len__(self):
        return self.x.shape[0]


print(x_train.shape,y_train.shape)

train_data = gtzandata(x_train,y_train)
valid_data = gtzandata(x_valid,y_valid)
test_data = gtzandata(x_test,y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last = True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

# model class
class model_1DCNN(nn.Module):
    def __init__(self):
        super(model_1DCNN, self).__init__()
   
        self.conv0 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8))

        self.conv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8))

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4))

        self.fc0 = nn.Linear(256, 64)
        self.fc1 = nn.Linear(64, 10)
        #self.activation = nn.Softmax()

    def forward(self,x):
        # input x: minibatch x 128 x 12XX

        #print(x.size())
        #x = x.view(x.size(0),x.size(2),x.size(1))
        # now x: minibatch x 12XX x 128

        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)

        #print(out.size())
        # flatten the out so that the fully connected layer can be connected from here
        out = out.view(x.size(0), out.size(1) * out.size(2))
        out = self.fc0(out)
        out = self.fc1(out)
        #out = self.activation(out)

        return out

# load model
if gpu_use == 1:
	model = model_1DCNN().cuda(which_gpu)
elif gpu_use == 0:
	model = model_1DCNN()

# training
criterion = nn.CrossEntropyLoss()

# train / eval
def fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        acc = []

        for i, data in enumerate(train_loader):
            audio = data['mel']
            label = data['label']
            # have to convert to an autograd.Variable type in order to keep track of the gradient...
            if gpu_use == 1:
                audio = Variable(audio).type(torch.FloatTensor).cuda(which_gpu)
                label = Variable(label).type(torch.LongTensor).cuda(which_gpu)
            elif gpu_use == 0:
                audio = Variable(audio).type(torch.FloatTensor)
                label = Variable(label).type(torch.LongTensor)
	
            optimizer.zero_grad()
            outputs = model(audio)

            #print(outputs,label)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ("Epoch [%d/%d], Iter [%d/%d] loss : %.4f" % (epoch+1, num_epochs, i+1, len(train_loader), loss.data[0]))

        eval_loss, _ , _ = eval(model, valid_loader, criterion)
        scheduler.step(eval_loss) # use the learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print('Learning rate : {}'.format(curr_lr))
        if curr_lr < 1e-5:
            print ("Early stopping\n\n")
            break

def eval(model,valid_loader,criterion):

    eval_loss = 0.0
    output_all = []
    label_all = []

    model.eval()
    for i, data in enumerate(valid_loader):
        audio = data['mel']
        label = data['label']
        # have to convert to an autograd.Variable type in order to keep track of the gradient...
        if gpu_use == 1:
            audio = Variable(audio).type(torch.FloatTensor).cuda(which_gpu)
            label = Variable(label).type(torch.LongTensor).cuda(which_gpu)
        elif gpu_use == 0:
            audio = Variable(audio).type(torch.FloatTensor)
            label = Variable(label).type(torch.LongTensor)
	
        outputs = model(audio)
        loss = criterion(outputs, label)

        eval_loss += loss.data[0]

        output_all.append(outputs.data.cpu().numpy())
        label_all.append(label.data.cpu().numpy())

    avg_loss = eval_loss/len(valid_loader)
    print ('Average loss: {:.4f} \n'. format(avg_loss))



    return avg_loss, output_all, label_all

# run
start_time = time.time()
fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs)
print("--- %s seconds ---" % (time.time() - start_time))


# evaluation
avg_loss, output_all, label_all = eval(model,test_loader,criterion)
#print(len(output_all),output_all[0].shape,avg_loss)

prediction = np.concatenate(output_all)
prediction = prediction.reshape(len(test_list),len(genres))
prediction = prediction.argmax(axis=1)
#print(prediction)

y_label = np.concatenate(label_all)
#print(y_label)

comparison = prediction - y_label
acc = float(len(test_list) - np.count_nonzero(comparison)) / len(test_list)
print('Test Accuracy: {:.4f} \n'. format(acc))

# TODO segmentation eval function average !!!







