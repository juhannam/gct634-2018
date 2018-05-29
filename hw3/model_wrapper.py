'''
model_wrapper.py

A file related with torch and running neural network model.
This code runs model from 'model_archive'.
You can change model, loss function and optimizer here.
'''
import os
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable

import model_archive

class Wrapper(object):
    #Settings of model, loss function, optimizer and scheduler
    def __init__(self, input_size, num_class, learn_rate):
        super(Wrapper, self).__init__()
        self.model = model_archive.RNN(input_size, num_class)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learn_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=3, verbose=True)

    #Accuracy function works like criterion in pytorch
    def accuracy(self, prediction, reference):
        prediction = prediction.max(1)[1].type(torch.LongTensor)
        reference = reference.cpu()
        correct = (prediction == reference).sum().item()

        return correct/float(prediction.size(0))

    #Running model function for train, test and validation.
    def run_model(self, x, y, device=0, mode='train'):
        if mode == 'train': self.model.train()
        elif mode == 'eval': self.model.eval()

        output_list = []
        total_acc = 0
        total_loss = 0

        for batch in range(x.shape[0]):
            input = Variable(torch.from_numpy(x[batch:batch+1])).type(torch.FloatTensor)
            label = Variable(torch.LongTensor(y[batch]))
            if device > 0:
                input = input.cuda(device - 1)
                label = label.cuda(device - 1)
                self.model = self.model.cuda(device - 1)

            output = self.model(input)
            acc = self.accuracy(output, label)
            loss = self.criterion(output, label)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            output_list.append(output.cpu().data.numpy())
            total_acc += acc
            total_loss += loss.item()

        total_output = np.concatenate(output_list).argmax(axis=1)
        total_acc = total_acc/x.shape[0]
        total_loss = total_loss/x.shape[0]

        return total_output, total_acc, total_loss

    #Early stopping function from validation loss
    def early_stop(self, loss):
        self.scheduler.step(loss)
        current_lr = self.optimizer.param_groups[0]['lr']
        print('Learning Rate : ' + str(round(current_lr,4)))
        stop = (current_lr < 1e-5)
        if stop:
            print('Early stopping\n\n')

        return stop

    #Function for saving trained model, annotation and predictionin given directory
    def export(self, export_dir, chroma, annotation, prediction):
        if not os.path.exists(os.path.dirname(export_dir)):
            os.makedirs(os.path.dirname(export_dir))

        model_save_path = os.path.join(export_dir, 'model.pth')
        torch.save(self.model, model_save_path)

        np.save(os.path.join(export_dir, 'chroma.npy'), chroma)
        np.save(os.path.join(export_dir, 'annotation.npy'), annotation)
        np.save(os.path.join(export_dir, 'prediction.npy'), prediction)
