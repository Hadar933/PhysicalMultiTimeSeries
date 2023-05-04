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
    train_percent, val_percent = 0.9, 0.09
    feature_win, target_win, intersect = 120, 1, 0
    batch_size = 64
    criterion = nn.L1Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience, patience_tolerance = 10, 0.005
    n_epochs = 20
    seed = 3407
    optimizer = torch.optim.Adam(model.parameters())
    features_norm = 'identity'
    targets_norm = 'minmax'
    model_name = f"lstm_fc_{features_norm}_input_and_{targets_norm}_targets"
    trainer = Trainer(forces, kinematics, train_percent, val_percent, feature_win, target_win, intersect, batch_size,
                      model, model_name, optimizer, criterion, device, patience, patience_tolerance, n_epochs, seed,
                      features_norm, targets_norm)
    trainer.load_trained_model('G:\My Drive\Master\Lab\Experiment\MultiTimeSeries\Models\lstm_fc_identity_input_and_minmax_targets_2023-04-29_10-39-24\\best_lstm_fc_identity_input_and_minmax_targets_2023-04-29_10-39-24_epoch_19.pt')
    # trainer.fit()
    ret = trainer.predict()
    ret = utils.format_df_torch_entries(ret)
    utils.plot(ret)
