from datetime import datetime
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import psutil
import torch
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torchsummary import summary
import torch.nn as nn
from datasets import ds
TIME_FORMAT = '%Y-%m-%d_%H-%M-%S'


class Trainer:
    """
    a class that takes in a model and some training parameters and performs testing, evaluating and prediction
    """

    def __init__(self,
                 model: torch.nn.Module,
                 model_name: str,
                 optimizer: torch.optim.Optimizer,
                 criterion: Optional[torch.nn.modules.loss] = torch.nn.MSELoss(),
                 device: Optional[torch.cuda.device] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 patience_tolerance: float = 0.005,
                 n_epochs: Optional[int] = 100,
                 patience: Optional[int] = 10,
                 batch_size: Optional[int] = 64,
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
        self.batch_size: int = batch_size
        self.stop_training: bool = False
        self.tb_writer: SummaryWriter = None
        self.best_val_loss: float = float('inf')
        self.model_dir: str = ""
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
            tqdm_loader.set_postfix({'Train Loss': f"{loss.item():.3f}",
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
                tqdm_loader.set_postfix({'Val Loss': f"{loss.item():.3f}",
                                         'RAM_%': psutil.virtual_memory().percent,
                                         'Epoch': epoch})
            total_loss /= len(val_loader)
        return total_loss

    def _finish_epoch(self, val_loss, prev_val_loss, time, epoch):
        if val_loss < self.best_val_loss:  # if val is improved, we update the best model
            self.best_val_loss = val_loss
            torch.save(self.model.state_dict(), f"{self.model_dir}/best_{self.model_name}_{time}_epoch_{epoch}.pt")
        elif torch.abs(val_loss - prev_val_loss) <= self.patience_tolerance:  # if val doesn't improve, counter += 1
            self.early_stopping += 1
        if self.early_stopping >= self.patience:  # if patience value is reached, the training process halts
            self.stop_training = True
            print(f"[Early Stopping]")

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
        predictions = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs.to(self.device)
                pred = self.model(inputs)
                predictions.append(pred)
        return torch.cat(predictions, 0)

    def load_trained_model(self, trained_model_path: str):
        self.model.load_state_dict(torch.load(trained_model_path))
        self.model.eval()


if __name__ == '__main__':
    mlp = torch.nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),
        nn.Linear(294, 16),
        nn.ReLU(),
        nn.Linear(16, 10),
        nn.Softmax(dim=1)
    )
    model_name = 'mlp'
    opt = torch.optim.Adam(mlp.parameters())
    crit = torch.nn.CrossEntropyLoss()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    tr_loader = torch.utils.data.DataLoader(dataset1, batch_size=64)
    te_loader = torch.utils.data.DataLoader(dataset2, batch_size=64)
    print(summary(mlp, (1, 28, 28)))
    T = Trainer(mlp, model_name, opt, crit)

    trained_model = "G:\\My Drive\\Master\\Lab\\Experiment\\Trainer\\Models\\mlp_2023-03-17_19-05-10\\best_mlp_2023-03-17_19-05-10_epoch"
    T.load_trained_model(trained_model)
    preds = T.predict(te_loader)
    x = 2
