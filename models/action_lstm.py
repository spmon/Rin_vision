# models/action_lstm.py
import torch
import torch.nn as nn
import config 

class LSTMActionModel(nn.Module):
    def __init__(self, input_size=34, hidden_size=32, num_layers=1, num_classes=len(config.LABEL_MAP_ACTION)):
        super(LSTMActionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.7)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(config.DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out