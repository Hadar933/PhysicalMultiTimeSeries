from typing import Tuple
from torch.utils.data import Dataset
import torch


class MultiTimeSeries(Dataset):
	def __init__(self, kinematics: torch.Tensor, forces: torch.Tensor,
				 feature_lags: int, target_lags: int, feature_target_intersection: int):
		"""
		denote FL as feature lag, TL as target lag, intersection as I and history size as H. Assume we want to predict
		at time t+FL+1, the time windows we provide are:
		
				ALL WINDOW: [0, 1, 2, 3,  ..., t-1, t, t+1,           ...               H-4, H-3, H-2, H-1]
				FEATURE WINDOW:                    [t, t+1, ..., t-FL-1, t+FL]
				TARGET WINDOW:                               [t+FL-I ,...t+FL, t+FL+1 , ..., t+FL+TL]
		
		as per the number of windows per ds - for every t we create a window, but also leaving room for the first
		window (FL timestamps) and the last window (TL timestamps), minus their intersection + 1 (cut t_0=0)
	
		:param kinematics: a tensor with shape (N,H,F), where:
					     - N: number of datasets
					     - H: number of samples per dataset
					     - F: number of features per sample
		:param forces: a tensor with shape (N,H,T) where T is the number of targets to predict
					   (currently supporting T=1)
		:param feature_lags: feature history to consider
		:param target_lags: target future to predict
		:param feature_target_intersection: intersection between target and features
		"""
		self.feature_lag: int = feature_lags
		self.target_lag: int = target_lags
		self.feature_target_intersect: int = feature_target_intersection
		self.kinematics: torch.Tensor = kinematics
		self.forces: torch.Tensor = forces
		self.n_datasets, self.n_samples_per_ds, self.n_features = self.kinematics.shape
		self.n_windows_per_ds = self.n_samples_per_ds - self.feature_lag + self.feature_target_intersect - self.target_lag + 1
	
	def __len__(self) -> int:
		return self.n_datasets * self.n_windows_per_ds
	
	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		# TODO: maybe add random start index.
		# print("====================================================================================================")
		ds_idx = idx // self.n_windows_per_ds
		win_idx = idx % self.n_windows_per_ds
		# print(f"idx={idx} [ds={ds_idx}, win={win_idx}]")
		features_window = self.kinematics[ds_idx, win_idx: win_idx + self.feature_lag]
		# print(f"feature window [{ds_idx}, {win_idx}:{win_idx + self.feature_lag}]-")
		# print(features_window)
		target_window = self.forces[ds_idx, win_idx + self.feature_lag - self.feature_target_intersect:
											win_idx + self.feature_lag - self.feature_target_intersect + self.target_lag]
		# print(f"target window [{ds_idx}, {win_idx + self.feature_lag - self.feature_target_intersect}:{win_idx + self.feature_lag - self.feature_target_intersect + self.target_lag}]-")
		# print(target_window)
		
		return features_window, target_window


if __name__ == '__main__':
	k = torch.Tensor([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
					  [[8, 8], [9, 9], [10, 10], [11, 11], [12, 12]]])
	
	f = torch.Tensor([[[1], [2], [3], [4], [5]],
					  [[8], [9], [10], [11], [12]]])
	ds = MultiTimeSeries(k, f, 3, 2, 1)
	dl = torch.utils.data.DataLoader(ds)
	print(len(ds))
	for x, y in dl:
		z = 2
