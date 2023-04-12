from typing import Optional, List

import pandas as pd

from trainer import Trainer
import matplotlib.pyplot as plt
import utils
import torch
import torch.nn as nn
from torchsummary import summary


def main(features: torch.Tensor,
         targets: torch.Tensor,
         model: nn.Module,
         model_name: str,
         feature_lag: Optional[int] = 120,
         target_lag: Optional[int] = 1,
         intersect: Optional[int] = 0,
         train_percent: Optional[float] = 0.9,
         val_percent: Optional[float] = 0.09,
         batch_size: Optional[int] = 64,
         criterion: Optional[torch.nn.modules.loss._Loss] = torch.nn.L1Loss(),
         device: Optional[torch.cuda.device] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
         patience_tolerance: Optional[float] = 0.005,
         n_epochs: Optional[int] = 100,
         patience: Optional[int] = 10,
         seed: Optional[int] = 3407):
    n_features = features.shape[-1]
    # print(summary(model, (1, feature_lag, n_features)))
    optimizer = torch.optim.Adam(model.parameters())
    train_loader, val_loader, test_loaders = utils.train_val_test_split(features, targets, train_percent, val_percent,
                                                                        feature_lag, target_lag, intersect, batch_size)
    trainer = Trainer(train_loader, val_loader, test_loaders, model, model_name, optimizer, criterion, device,
                      patience, patience_tolerance, n_epochs, seed)
    trainer.load_trained_model(
        "G:\My Drive\Master\Lab\Experiment\ForcePrediction\Models\lstm_2023-04-12_17-08-28/best_lstm_2023-04-12_17-08-28_epoch_0.pt")
    # trainer.fit()
    predictions = trainer.predict()
    return {'trainer': trainer,
            'predictions': predictions}


if __name__ == '__main__':
    # 64 x 120 x 7
    mlp_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(840, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    )


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


    lstm_model = LSTM()
    mod_name = "lstm"
    kinematics, forces = utils.load_data_from_prssm_paper(kinematics_key='ds_u')

    ret = main(kinematics, forces, lstm_model, mod_name, n_epochs=1)
    x = 2
