from typing import Optional, Union, Dict, Tuple
import scipy
from plotly_resampler import FigureResampler
import pandas as pd
import plotly.graph_objects as go
import torch

from ForcePrediction.datasets import MultipleTS


def tensor_mb_size(v: torch.Tensor):
	return v.nelement() * v.element_size() / 1_000_000


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


def read_data(data_path) -> pd.DataFrame:
	"""	reads a gzip file with possible 'ts' column and saves the df as the preprocessed data. """
	try: data = pd.read_csv(data_path, index_col=0, parse_dates=['ts'], compression="gzip")
	except ValueError: data = pd.read_csv(data_path, index_col=0, compression="gzip")
	return data


def load_data_from_prssm_paper(path: str = "G:\\My Drive\\Master\\Lab\\flapping-wing-aerodynamics-prssm\\Data\\Input\\"
										   "flapping_wing_aerodynamics.mat", generate_dummy: bool = True) -> Union[
	Dict, Tuple[torch.Tensor, torch.Tensor]]:
	mat = scipy.io.loadmat(path)
	if generate_dummy:
		kinematics = torch.Tensor(mat['ds_pos'])  # stroke, deviation, rotation
		forces = torch.Tensor(mat['ds_y_raw'])[:, :, 0]  # F_normal
		return kinematics, forces
	return mat


def train_val_test_split(kinematics, forces, train_percent: float = 0.90, val_percent: float = 0.09,
						 feature_lag: int = 120, target_lag: int = 0, intersect: int = 0, batch_size: int = 64):
	n_exp, hist_size, n_features = kinematics.shape
	train_size = int(train_percent * n_exp)
	val_size = int(val_percent * n_exp)
	kinematics_train, forces_train = kinematics[:train_size], forces[:train_size]
	kinematics_val, forces_val = kinematics[train_size:train_size + val_size], forces[train_size:train_size + val_size]
	kinematics_test, forces_test = kinematics[train_size + val_size:], forces[train_size + val_size:]
	train_dataset = MultipleTS(kinematics_train, forces_train, feature_lag, target_lag, intersect)
	val_dataset = MultipleTS(kinematics_val, forces_val, feature_lag, target_lag, intersect)
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	# since a prediction should be on a single dataset, we create a dataset (and a dataloader) for each of
	# the test datasets in the test tensor. In inference time, we can simply choose the test dataloader
	# using its index.
	all_test_datasets = {
		i: MultipleTS(kinematics_test[i].unsqueeze(0), forces_test[i].unsqueeze(0), feature_lag, target_lag, intersect) for i in
		range(kinematics_test.shape[0])
	}
	all_test_dataloaders = {
		i: torch.utils.data.DataLoader(all_test_datasets[i], batch_size=1, shuffle=False) for i in
		range(len(all_test_datasets))
	}
	return train_loader, val_loader, all_test_dataloaders


if __name__ == '__main__':
	k, f = load_data_from_prssm_paper()
