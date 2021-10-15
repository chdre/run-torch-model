import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class RunTorchModel:
    """

    """

    def __init__(self, model, epochs, optimizer, dataloaders, criterion):
        self.epochs = epochs
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if len(dataloaders) >= 2:
            self.dataloader_train = dataloaders[0]
            self.dataloader_test = dataloaders[1]
            if len(dataloaders) == 3:
                self.dataloader_valid = dataloaders[2]

    def run_epoch(self, dataloader):
        all_targets = dataloader.dataset[:][1].to(self.device)

        if self.training:
            self.model.train()
        else:
            self.model.eval()

        _predictions = []
        total_loss = torch.zeros(1, device=torch.device(self.device))

        for batch_idx, data_batch in enumerate(dataloader):
            features, targets = data_batch[0].to(
                self.device), data_batch[1].to(self.device)

            if self.training:
                self.train(features, targets)
            elif not self.training:
                self.mtest(features, targets)

            total_loss += self.batch_loss
            _predictions.append(self.batch_predictions)

        self.predictions = torch.cat(_predictions, dim=0)
        self.loss_avg = total_loss.item() / len(dataloader)
        self.r2 = 1 - self.criterion(self.predictions,
                                     all_targets) / torch.var(all_targets)

    def __call__(self):
        r2score = torch.zeros(self.epochs).to(self.device)
        loss_avg = torch.zeros(self.epochs).to(self.device)
        r2score_test = torch.zeros(self.epochs).to(self.device)
        loss_avg_test = torch.zeros(self.epochs).to(self.device)

        for i in range(self.epochs):
            self.training = True
            self.run_epoch(self.dataloader_train)
            r2score[i] = self.r2
            loss_avg[i] = self.loss_avg

            self.training = False
            self.run_epoch(self.dataloader_test)
            r2score_test[i] = self.r2
            loss_avg_test[i] = self.loss_avg

    def get_predictions(self):
        return self.predictions

    def get_average_loss(self):
        return self.loss_avg

    def get_r2score(self):
        return self.r2.item()

    def train(self, features, targets):
        """Train run for model.

        :param features: self explanatory
        :type features: torch.Tensor
        :param targets: self explanatory
        :type targets: torch.Tensor
        """
        self.batch_predictions = self.model(features)
        self.batch_loss = self.criterion(self.batch_predictions, targets)

        self.optimizer.zero_grad()
        # below is ???
        # for param in model.parameters():
        #         param.grad = None
        self.batch_loss.backward()
        self.optimizer.step()

    def mtest(self, features, targets):
        """Testing run for model.

        :param features: self explanatory
        :type features: torch.Tensor
        :param targets: self explanatory
        :type targets: torch.Tensor
        """
        with torch.no_grad():
            self.batch_predictions = self.model(features)
            self.batch_loss = self.criterion(self.batch_predictions, targets)

    def evaluate(self, dataloader):
        """Typically used for evaluating the model by validation set.
        :param dataloader: Dataset to evaluate the model with.
        :type dataloader: torch.dataloader
        """
        features = dataloader.dataset[:][0].to(self.device)
        targets = dataloader.dataset[:][1].to(self.device)
        with torch.no_grad():
            predictions = model(features)
            loss = self.criterion(predictions, targets)

        r2 = 1 - self.criterion(predictions, targets) / torch.var(targets)

        return predictions, loss

    def predict(self, features):
        """Use the trained model to perform predictions on unseen data.
        """
        with torch.no_grad():
            predictions = model(features)

        return predictions
