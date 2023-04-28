import torch
import torch.nn as nn
from typing import List, Tuple


class LSTM(nn.Module):
    """ Takes only the last output of the LSTM and propagates it to a FC layer """

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x.unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, history_size: int, output_dim: int, hidden_dims_list: List[int]):
        """
        an MLP implementation that takes performs (Batch,Hist,Features) -> (Batch,Hist*Features) -> MLP -> (Batch,Out)
        :param input_dim: the number of features in the input, required to calculate the MLP input dim
        :param history_size: feature window, essentially, required to calculate the MLP input dim
        :param output_dim: output dim for multi-target prediction
        :param hidden_dims_list: the dimensions of the hidden layers only (without input and output)
        """
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim * history_size, hidden_dims_list[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims_list) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims_list[i], hidden_dims_list[i + 1]))
            self.hidden_layers.append(nn.ReLU())
        self.output_layer = nn.Linear(hidden_dims_list[-1], output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
