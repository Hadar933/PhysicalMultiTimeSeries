from typing import Optional
from plotly_resampler import FigureResampler
import pandas as pd
import plotly.graph_objects as go
import torch


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
