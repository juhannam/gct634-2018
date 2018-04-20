# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import lr_scheduler

# train / eval
def fit(model,train_loader,valid_loader,criterion,learning_rate,num_epochs, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, nesterov=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        acc = []

        for i, data in enumerate(train_loader):
            audio = data['mel']
            label = data['label']
            # have to convert to an autograd.Variable type in order to keep track of the gradient...
            if args.gpu_use == 1:
                audio = Variable(audio).type(torch.FloatTensor).cuda(args.which_gpu)
                label = Variable(label).type(torch.LongTensor).cuda(args.which_gpu)
            elif args.gpu_use == 0:
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

        eval_loss, _ , _ = eval(model, valid_loader, criterion,args)
        scheduler.step(eval_loss) # use the learning rate scheduler
        curr_lr = optimizer.param_groups[0]['lr']
        print('Learning rate : {}'.format(curr_lr))
        if curr_lr < 1e-5:
            print ("Early stopping\n\n")
            break

def eval(model,valid_loader,criterion, args):

    eval_loss = 0.0
    output_all = []
    label_all = []

    model.eval()
    for i, data in enumerate(valid_loader):
        audio = data['mel']
        label = data['label']
        # have to convert to an autograd.Variable type in order to keep track of the gradient...
        if args.gpu_use == 1:
            audio = Variable(audio).type(torch.FloatTensor).cuda(args.which_gpu)
            label = Variable(label).type(torch.LongTensor).cuda(args.which_gpu)
        elif args.gpu_use == 0:
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






