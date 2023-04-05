from trainer import Trainer
import matplotlib.pyplot as plt
from utils import load_data_from_prssm_paper, train_val_test_split
import torch
import torch.nn as nn
from torchsummary import summary


def main():
	kinematics, forces = load_data_from_prssm_paper()
	feature_lag, target_lag, intersect = 120, 1, 0
	train_percent, val_percent = 0.9, 0.09
	batch_size = 64
	n_features = kinematics.shape[-1]
	# input: (64,120,3)
	model = nn.Sequential()
	print(summary(model, (1, feature_lag, n_features)))
	optimizer = torch.optim.Adam(model.parameters())
	trainer = Trainer(model, 'linear', optimizer)
	train_loader, val_loader, all_test_dataloaders = train_val_test_split(kinematics, forces, train_percent, val_percent,
		feature_lag, target_lag, intersect, batch_size)
	trainer.fit(train_loader, val_loader)
	for i in range(len(all_test_dataloaders)):
		preds, trues = trainer.predict(all_test_dataloaders[i])
		plt.plot(preds, label="Predictions")
		plt.plot(trues, label="True Labels")
		plt.legend()
		plt.title(f"Test set #{i}, MAE={nn.functional.l1_loss(preds, trues)}")
		plt.show()


if __name__ == '__main__':
	main()
