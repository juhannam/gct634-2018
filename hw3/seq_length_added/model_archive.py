'''
model_archive.py

A file that contains neural network models.
You can also make different model like CNN if you follow similar format like given RNN.
'''
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, 1, bidirectional=True, dropout=0.25)
        self.linear = nn.Linear(2*64, num_classes)  # 2 for bidirection

    def forward(self, x):
        output, hidden = self.lstm(x, None)
        output = self.linear(output[-1])

        return output
