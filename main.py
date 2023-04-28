from typing import Optional, List
import ModelZoo as zoo
import pandas as pd

from MultiTimeSeries.normalizer import Normalizer
from trainer import Trainer
import matplotlib.pyplot as plt
import utils
import torch
import torch.nn as nn
from torchsummary import summary

if __name__ == '__main__':
    # 64 x 120 x 5
    kinematics, forces = utils.load_data_from_prssm_paper()
    input_dim, output_dim = forces.shape[-1], kinematics.shape[-1]
    hidden_dim, num_lstms = 32, 1
    model = zoo.LSTM(input_dim, hidden_dim, num_lstms, output_dim)
    model_name = "lstm_fc_minmax_input"
    train_percent, val_percent = 0.9, 0.09
    feature_win, target_win, intersect = 120, 1, 0
    batch_size = 64
    criterion = nn.L1Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience, patience_tolerance = 10, 0.005
    n_epochs = 20
    seed = 3407
    # model = zoo.MLP(forces.shape[-1], feature_win, kinematics.shape[-1], [10, 20, 30])
    optimizer = torch.optim.Adam(model.parameters())
    features_norm = 'minmax'
    targets_norm = 'minmax'
    trainer = Trainer(forces, kinematics, train_percent, val_percent, feature_win, target_win, intersect, batch_size,
                      model, model_name, optimizer, criterion, device, patience, patience_tolerance, n_epochs, seed,
                      features_norm, targets_norm)
    trainer.fit()
    ret = trainer.predict()
    ret = utils.format_df_torch_entries(ret)
    utils.plot(ret)
