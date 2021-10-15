import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class RunTorchCNN:
    """Simple package to execute training for a pytorch compatible CNN.

    :param model: Pytorch model
    :type model: torch.nn.Module
    :param epochs: Number of epochs to train, i.e. number of training iterations
    :type epochs: int
    :param optimizer: Pytorch optimizer, e.g. torch.optim.Adam
    :type optimizer: torch.optim
    :param dataloaders: Datasets containing features and targets
    :type dataloaders: tuple of type torch.data.dataloader
    :param criterion: Loss function for NN
    :type criterion: torch.nn.module.loss
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
        self.r2score_train = torch.zeros(self.epochs).to(self.device)
        self.loss_avg_train = torch.zeros(self.epochs).to(self.device)
        self.r2score_test = torch.zeros(self.epochs).to(self.device)
        self.loss_avg_test = torch.zeros(self.epochs).to(self.device)

        for i in range(self.epochs):
            self.training = True
            self.run_epoch(self.dataloader_train)
            self.r2score_train[i] = self.r2
            self.loss_avg_train[i] = self.loss_avg

            self.training = False
            self.run_epoch(self.dataloader_test)
            self.r2score_test[i] = self.r2
            self.loss_avg_test[i] = self.loss_avg

    def get_predictions(self):
        """Returns predictions. Function could return predictions from
        training, testing or validation, show care when using. Will most likely
        be removed as it is kind of useless.
        """
        return self.predictions

    def get_growing_loss(self):
        """Returns train and test loss as a function of epochs."""
        return self.loss_avg_train, self.loss_avg_test

    def get_growing_r2(self):
        """Returns train and test R2 score as a function of epochs."""
        return self.r2score_train, self.r2score_test

    def get_average_loss(self):
        """Returns average loss."""
        return self.loss

    def get_r2score(self):
        """Returns R2 score."""
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
        :returns predictions, loss, r2: self explanatory
        :rtype: torch.Tensor, float, float
        """
        features = dataloader.dataset[:][0].to(self.device)
        targets = dataloader.dataset[:][1].to(self.device)
        with torch.no_grad():
            predictions = model(features)
            loss = self.criterion(predictions, targets)

        r2 = 1 - self.criterion(predictions, targets) / torch.var(targets)

        return loss, r2

    def predict(self, features):
        """Use the trained model to perform predictions on unseen data.
        :param features: A set of features which are compatible with the model
        :type features: torch.Tensor
        :returns predictions: self explanatory
        :rtype predictions: torch.Tensor
        """
        with torch.no_grad():
            predictions = model(features)

        return predictions
