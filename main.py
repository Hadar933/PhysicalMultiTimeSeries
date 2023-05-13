from typing import Optional, List

from MultiTimeSeries.Zoo import seq2seq
from Zoo import rnn
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
    dropout = 0.1
    bidirectional = True

    model = rnn.RNN('lstm', input_dim, output_dim, hidden_dim, num_lstms, dropout, bidirectional)
    train_percent, val_percent = 0.9, 0.09
    feature_win, target_win, intersect = 120, 1, 0
    batch_size = 64

    # model = seq2seq.Seq2Seq((batch_size, feature_win, input_dim), target_win, 10, hidden_dim, 1, bidirectional,
    #                         10, hidden_dim, output_dim)
    criterion = nn.L1Loss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patience, patience_tolerance = 10, 0.005
    n_epochs = 30
    seed = 3407
    optimizer = torch.optim.Adam(model.parameters())
    features_norm = 'identity'
    targets_norm = 'zscore'
    features_global_norm = True
    targets_global_norm = True
    model_name = f"bilstm_fc_{features_norm}_input_and_{targets_norm}"

    trainer = Trainer(forces, kinematics, train_percent, val_percent, feature_win, target_win, intersect, batch_size,
                      model, model_name, optimizer, criterion, device, patience, patience_tolerance, n_epochs, seed,
                      features_norm, targets_norm, features_global_norm, targets_global_norm)
    # TODO: when loading a model, all trainer arguments should be loaded as well !
    # trainer.load_trained_model(
    #     "G:\\My Drive\\Master\\Lab\\Experiment\\MultiTimeSeries\\Models\\lstm_fc_identity_input_and_minmax_targets_2023-05-07_11-58-39/best_lstm_fc_identity_input_and_minmax_targets_2023-05-07_11-58-39_epoch_19.pt")
    trainer.fit()
    ret = trainer.predict()
    ret = utils.format_df_torch_entries(ret)
    utils.plot(ret)
    x = 2
