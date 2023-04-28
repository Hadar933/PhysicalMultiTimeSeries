from typing import Dict

import pandas as pd
import torch

from MultiTimeSeries import utils


class Normalizer:
    def __init__(self, method: str = 'zscore', history_dim: int = 1):
        self.method: str = method
        self.history_dim: int = history_dim
        self.normalization_params: Dict = {}

    def fit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        normalizes (feature-wise) a tensor with shape (N,H,F) where
            - N is the number of experiments
            - H is the history of every experiment
            - F is the number of features in every experiment
        :return: tensor t such that for every exp i=0,1,2,...,N-1, the matrix t[i,:,:] columns are normalized
        """
        if self.method == 'zscore':
            return self.zscore(tensor)
        elif self.method == 'minmax':
            return self.minmax(tensor)
        elif self.method == 'identity':
            return tensor
        else:
            raise ValueError(f"Normalization method '{self.method}' not recognized.")

    def zscore(self, tensor: torch.Tensor) -> torch.Tensor:
        """ performs zscore normalization on the tensor """
        mean_val = tensor.mean(self.history_dim, keepdim=True)
        std_val = tensor.std(self.history_dim, keepdim=True)
        self.normalization_params = {'mean': mean_val, 'std': std_val}
        norm_tensor = (tensor - mean_val) / std_val
        return norm_tensor

    def minmax(self, tensor: torch.Tensor) -> torch.Tensor:
        """ performs minmax normalization on the tensor """
        min_val = tensor.min(self.history_dim, keepdim=True).values
        max_val = tensor.max(self.history_dim, keepdim=True).values
        self.normalization_params = {'min': min_val, 'max': max_val}
        norm_tensor = (tensor - min_val) / (max_val - min_val)
        return norm_tensor


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()  # for testing purposes
    kinematics, forces = utils.load_data_from_prssm_paper()
    N = Normalizer()
    n_forces = N.fit_transform(forces)
    exp = 0
    f, fn = forces[exp, :, :], n_forces[exp, :, :]
    fsk = torch.from_numpy(scaler.fit_transform(f.numpy()))
    df_f = pd.DataFrame(f, columns=[f"f_{i}" for i in range(f.shape[-1])])
    df_fn = pd.DataFrame(fn, columns=[f"fn_{i}" for i in range(fn.shape[-1])])
    df_fsk = pd.DataFrame(fsk, columns=[f"fsk_{i}" for i in range(fsk.shape[-1])])
    df = pd.concat([df_f, df_fn, df_fsk], axis=1)
    utils.plot(df)
