from datetime import datetime
from typing import Optional, List, Tuple, Dict
import utils
import pandas as pd
import yaml
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
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 all_test_loaders: List[torch.utils.data.DataLoader],
                 model: torch.nn.Module,
                 model_name: str,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.modules.loss._Loss,
                 device: torch.cuda.device,
                 patience: int,
                 patience_tolerance: float,
                 n_epochs: int,
                 seed: int):
        torch.manual_seed(seed)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.all_test_loaders = all_test_loaders
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
        self.yaml_path = ""
        self._create_model_dir()

    def _create_model_dir(self):
        """ called when a trainer is initialized and creates a model dir with relevant information txt file(s) """
        init_timestamp = datetime.now().strftime(TIME_FORMAT)
        self.model_dir = f"{os.getcwd()}\\Models\\{self.model_name}_{init_timestamp}"
        self.yaml_path = f"{self.model_dir}\\trainer_info.yaml"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        with open(self.yaml_path, 'w') as f: yaml.dump({}, f)  # creating an empty args.yaml
        utils.update_yaml(self.yaml_path, {'model': str(self.model), 'patience': self.patience,
                                           'patience_tolerance': self.patience_tolerance, 'loss': str(self.criterion),
                                           'optim': str(self.optimizer), 'epochs': self.n_epochs})

    def _train_one_epoch(self, epoch: int) -> float:
        """ performs a training process for one epoch """
        self.model.train(True)
        total_loss = 0.0
        tqdm_loader = tqdm(self.train_loader)
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
        total_loss /= len(self.train_loader)
        return total_loss

    def _evaluate(self, epoch: int) -> float:
        """ evaluates the model on the validation set """
        self.model.train(False)
        total_loss = 0.0
        with torch.no_grad():
            tqdm_loader = tqdm(self.val_loader)
            for inputs, targets in tqdm_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                total_loss += loss
                tqdm_loader.set_postfix({'Val Loss': f"{loss.item():.5f}",
                                         'RAM_%': psutil.virtual_memory().percent,
                                         'Epoch': epoch})
            total_loss /= len(self.val_loader)
        return total_loss

    def _finish_epoch(self, val_loss, prev_val_loss, time, epoch):
        """ saves the best model and performs early stopping if needed """
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

    def fit(self) -> None:
        """ fits the model to the training data, with early stopping """
        fit_time = datetime.now().strftime(TIME_FORMAT)
        self.tb_writer = SummaryWriter(f'runs/{self.model_name}_{fit_time}')
        prev_val_loss = float('inf')
        for epoch in range(self.n_epochs):
            train_avg_loss = self._train_one_epoch(epoch)
            val_avg_loss = self._evaluate(epoch)
            self.tb_writer.add_scalar('Loss/Train', train_avg_loss, epoch)
            self.tb_writer.add_scalar('Loss/Val', val_avg_loss, epoch)
            self._finish_epoch(val_avg_loss, prev_val_loss, fit_time, epoch)
            if self.stop_training:
                break
            prev_val_loss = val_avg_loss
        utils.update_yaml(self.yaml_path, {"early_stopping": self.early_stopping,
                                           "best_val_loss": self.best_val_loss.value(),
                                           "best_model_path": self.best_model_path})

    def predict(self) -> pd.DataFrame:
        """ creates a prediction for every test loader in our test loaders list """
        self.model.train(False)
        all_preds = pd.DataFrame()
        with torch.no_grad():
            for j, test_loader in enumerate(self.all_test_loaders):
                curr_preds, curr_trues = [], []
                for inputs_i, true_i in test_loader:
                    inputs_i.to(self.device)
                    pred_i = self.model(inputs_i)
                    curr_preds.append(pred_i.item())
                    curr_trues.append(true_i.item())
                all_preds[f"pred_{j}"] = curr_preds
                all_preds[f"true_{j}"] = curr_trues
        return all_preds

    def load_trained_model(self, trained_model_path: str) -> None:
        """ loads into the trainer a trained model from memory """
        self.model.load_state_dict(torch.load(trained_model_path))
        self.model.eval()
