from typing import Optional, List

import pandas as pd

from trainer import Trainer
import matplotlib.pyplot as plt
import utils
import torch
import torch.nn as nn
from torchsummary import summary


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(7, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 64 x 120 x 7
    model = LSTM()
    model_name = "lstm"
    kinematics, forces = utils.load_data_from_prssm_paper()
    train_percent, val_percent = 0.9, 0.09
    feature_lag, target_lag, intersect = 120, 1, 0
    batch_size = 64
    criterion = torch.nn.L1Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience, patience_tolerance = 10, 0.005
    n_epochs = 20
    seed = 3407

    # n_features = features.shape[-1]
    # print(summary(model, (1, feature_lag, n_features)))

    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(kinematics,forces, train_percent, val_percent, feature_lag, target_lag, intersect, batch_size,
                      model, model_name, optimizer, criterion, device, patience, patience_tolerance, n_epochs, seed)
    # trained_model_path = "G:\My Drive\Master\Lab\Experiment\MultiTimeSeries\Models\lstm_2023-04-12_17-08-28/" \
    #                      "best_lstm_2023-04-12_17-08-28_epoch_0.pt"
    # trainer.load_trained_model(trained_model_path)
    trainer.fit()
    ret = trainer.predict()

    utils.plot(ret['predictions'])
