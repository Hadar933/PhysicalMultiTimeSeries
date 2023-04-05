from datetime import datetime
from typing import Optional

import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import psutil
import torch
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torchsummary import summary
import torch.nn as nn

TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


class Trainer:
	"""
	a class that takes in a model and some training parameters and performs testing, evaluating and prediction
	"""
	
	def __init__(self,
				 model: torch.nn.Module,
				 model_name: str,
				 optimizer: torch.optim.Optimizer,
				 criterion: Optional[torch.nn.modules.loss._Loss] = torch.nn.MSELoss(),
				 device: Optional[torch.cuda.device] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
				 patience_tolerance: float = 0.005,
				 n_epochs: Optional[int] = 100,
				 patience: Optional[int] = 10,
				 seed: int = 3407):
		torch.manual_seed(seed)
		self.patience_tolerance: float = patience_tolerance
		self.early_stopping: int = 0
		self.model_name: str = model_name
		self.criterion: torch.nn.modules.loss = criterion
		self.optimizer: torch.optim.Optimizer = optimizer
		self.device: torch.cuda.device = device
		self.model: torch.nn.Module = model.to(device)
		self.n_epochs: int = n_epochs
		self.patience: int = patience
		self.stop_training: bool = False
		self.tb_writer: SummaryWriter = None
		self.best_val_loss: float = float('inf')
		self.model_dir: str = ""
		self.best_model_path: str = ""
		self._create_model_dir()
	
	def _create_model_dir(self):
		""" called when a trainer is initialized and creates a model dir with model architecture .txt file """
		init_timestamp = datetime.now().strftime(TIME_FORMAT)
		self.model_dir = f"{os.getcwd()}\\Models\\{self.model_name}_{init_timestamp}"
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		with open(f"{self.model_dir}/architecture", 'w') as f:
			f.write(str(self.model))
	
	def _train_one_epoch(self, epoch: int, train_loader: torch.utils.data.DataLoader) -> float:
		"""
		performs a training process for one epoch
		:param epoch: the current epoch number
		:param train_loader: the training data loader
		"""
		self.model.train(True)
		total_loss = 0.0
		tqdm_loader = tqdm(train_loader)
		for inputs, targets in tqdm_loader:
			self.optimizer.zero_grad()
			inputs, targets = inputs.to(self.device), targets.to(self.device)
			predictions = self.model(inputs)
			loss = self.criterion(predictions, targets)
			total_loss += loss
			loss.backward()
			self.optimizer.step()
			tqdm_loader.set_postfix({'Train Loss': f"{loss.item():.5f}",
									 'RAM_%': psutil.virtual_memory().percent,
									 'Epoch': epoch})
		total_loss /= len(train_loader)
		return total_loss
	
	def _evaluate(self, epoch: int, val_loader: torch.utils.data.DataLoader) -> float:
		self.model.train(False)
		total_loss = 0.0
		with torch.no_grad():
			tqdm_loader = tqdm(val_loader)
			for inputs, targets in tqdm_loader:
				inputs, targets = inputs.to(self.device), targets.to(self.device)
				predictions = self.model(inputs)
				loss = self.criterion(predictions, targets)
				total_loss += loss
				tqdm_loader.set_postfix({'Val Loss': f"{loss.item():.5f}",
										 'RAM_%': psutil.virtual_memory().percent,
										 'Epoch': epoch})
			total_loss /= len(val_loader)
		return total_loss
	
	def _finish_epoch(self, val_loss, prev_val_loss, time, epoch):
		if val_loss < self.best_val_loss:  # if val is improved, we update the best model
			self.best_val_loss = val_loss
			if self.best_model_path:
				os.remove(self.best_model_path)
			self.best_model_path = f"{self.model_dir}/best_{self.model_name}_{time}_epoch_{epoch}.pt"
			torch.save(self.model.state_dict(), self.best_model_path)
		elif torch.abs(val_loss - prev_val_loss) <= self.patience_tolerance:  # if val doesn't improve, counter += 1
			self.early_stopping += 1
		if self.early_stopping >= self.patience:  # if patience value is reached, the training process halts
			self.stop_training = True
			print(f"[Early Stopping] at epoch {epoch}.")
	
	def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> None:
		fit_time = datetime.now().strftime(TIME_FORMAT)
		self.tb_writer = SummaryWriter()
		prev_val_loss = float('inf')
		for epoch in range(self.n_epochs):
			train_avg_loss = self._train_one_epoch(epoch, train_loader)
			val_avg_loss = self._evaluate(epoch, val_loader)
			self.tb_writer.add_scalar('Loss/Train', train_avg_loss, epoch)
			self.tb_writer.add_scalar('Loss/Val', val_avg_loss, epoch)
			self._finish_epoch(val_avg_loss, prev_val_loss, fit_time, epoch)
			if self.stop_training:
				break
			prev_val_loss = val_avg_loss
	
	def predict(self, test_loader: torch.utils.data.DataLoader):
		self.model.train(False)
		predictions, true_labels = [], []
		with torch.no_grad():
			for inputs, true in test_loader:
				inputs.to(self.device)
				pred = self.model(inputs)
				predictions.append(pred.item())
				true_labels.append(true.item())
		return torch.Tensor(predictions), torch.Tensor(true_labels)
	
	def load_trained_model(self, trained_model_path: str):
		self.model.load_state_dict(torch.load(trained_model_path))
		self.model.eval()


