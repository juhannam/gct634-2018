# GCT634 (2018) HW2
#
# Apr-20-2018: refactored version
# 
# Jongpil Lee
#

from __future__ import print_function
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# data loader
class gtzandata(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __getitem__(self,index):

        mel = self.x[index]

        entry = {'mel': mel, 'label': self.y[index]}
        
        return entry

    def __len__(self):
        return self.x.shape[0]



