from typing import Optional, Union, Dict, Tuple, Literal, List

import numpy as np
import scipy
import yaml
from plotly_resampler import FigureResampler
import pandas as pd
import plotly.graph_objects as go
import torch

from ForcePrediction.datasets import MultiTimeSeries


def tensor_mb_size(v: torch.Tensor):
    return v.nelement() * v.element_size() / 1_000_000


def update_yaml(yaml_path: str, new_data: Dict):
    """ updates a given yaml file with new data """
    with open(yaml_path, 'a') as f:
        yaml.dump(new_data, f)


def plot(df: pd.DataFrame,
         title: Optional[str] = "Data vs Time",
         x_title: Optional[str] = "time",
         y_title: Optional[str] = "Features") -> None:
    """	plots a df with plotly resampler """
    fig = FigureResampler(go.Figure())
    for col in df.columns:
        fig.add_trace(go.Scattergl(name=col, showlegend=True), hf_x=df.index, hf_y=df[col])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.show_dash(mode='external')


def read_data(data_path: str) -> pd.DataFrame:
    """	reads a gzip file with possible 'ts' column and saves the df as the preprocessed data. """
    try:
        data = pd.read_csv(data_path, index_col=0, parse_dates=['ts'], compression="gzip")
    except ValueError:
        data = pd.read_csv(data_path, index_col=0, compression="gzip")
    return data


def load_data_from_prssm_paper(path: str = "G:\\My Drive\\Master\\Lab\\flapping-wing-aerodynamics-prssm\\Data\\Input\\"
                                           "flapping_wing_aerodynamics.mat",
                               kinematics_key: Optional[Literal['ds_pos', 'ds_u_raw', 'ds_u']] = "ds_pos",
                               forces_key: Optional[Literal['ds_y_raw', 'ds_y']] = "ds_y_raw",
                               return_all: Optional[bool] = False,
                               targets_to_take: Optional[List[int]] = None) -> \
        Union[Dict, Tuple[torch.Tensor, torch.Tensor]]:
    """
    loads the data from the PRSSM paper, based on a string that represents the request
    :param path: path to the .mat file
    :param kinematics_key: one of the following options:
                        - 'ds_pos' (default): raw stroke, deviation and rotation
                        - 'ds_u_raw': The 7 kinematic variables derived from ds_pos
                        - 'ds_u': The standardized* versions of ds_u_raw
    :param forces_key: one of the following options:
                        - 'ds_y_raw' (default): raw targets Fx (normal), Fy (chord-wise), Mx, My, Mz
                        - 'ds_y': the standardized* version of ds_y_raw
    :param return_all: if true, returns the entire data dictionary
    :param targets_to_take: the columns to consider as targets. If not specified, takes all of them (5)
    * beware - this is normalized w.r.t an unknown training data
    :return:
    """
    mat: Dict = scipy.io.loadmat(path)
    if return_all: return mat
    kinematics = torch.Tensor(mat[kinematics_key])
    forces = torch.Tensor(mat[forces_key])
    forces = forces[:, :, 0] if not targets_to_take else forces[:, :, targets_to_take]
    return kinematics, forces


def train_val_test_split(kinematics: np.ndarray, forces: np.ndarray,
                         train_percent: float, val_percent: float,
                         feature_lag: int, target_lag: int, intersect: int,
                         batch_size: int) -> Tuple:
    """
    creates a time series train-val-test split for multiple multivariate time series
    :param kinematics: the data itself
    :param forces: the target(s)
    :param train_percent: percentage of data to be considered as training data
    :param val_percent: same, just for validation
    :param feature_lag: the number of samples in every window (history size)
    :param target_lag: the number of samples in the predicted value (usually one)
    :param intersect: an intersection between the feature and target lags
    :param batch_size: the batch size for the training/validation sets.
    :return: for the training and validation - regular pytorch loader. for the test - a loader for every dataset.
    """
    n_exp, hist_size, n_features = kinematics.shape
    train_size = int(train_percent * n_exp)
    val_size = int(val_percent * n_exp)
    kinematics_train, forces_train = kinematics[:train_size], forces[:train_size]
    kinematics_val, forces_val = kinematics[train_size:train_size + val_size], forces[train_size:train_size + val_size]
    kinematics_test, forces_test = kinematics[train_size + val_size:], forces[train_size + val_size:]
    train_dataset = MultiTimeSeries(kinematics_train, forces_train, feature_lag, target_lag, intersect)
    val_dataset = MultiTimeSeries(kinematics_val, forces_val, feature_lag, target_lag, intersect)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # since a prediction should be on a single dataset, we create a dataset (and a dataloader) for each of
    # the test datasets in the test tensor. In inference time, we can simply choose the test dataloader
    # using its index.
    all_test_datasets = [MultiTimeSeries(
        kinematics_test[i].unsqueeze(0),
        forces_test[i].unsqueeze(0),
        feature_lag, target_lag, intersect
    ) for i in range(kinematics_test.shape[0])
    ]
    all_test_dataloaders = [torch.utils.data.DataLoader(
        all_test_datasets[i],
        batch_size=1,
        shuffle=False
    ) for i in range(len(all_test_datasets))
    ]
    return train_loader, val_loader, all_test_dataloaders
