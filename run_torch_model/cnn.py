import torch
from torch import nn
import torch.nn.functional as F


class RunTorchModel:
    """

    """

    def __init__(self, model, epochs, optimizer, dataloader, criterion):
        self.epochs = epochs
        self.optimizer = optimizer
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.training = False

        self.targets = dataloader.dataset[:][1].to(self.device)

        # Variance of targets. Used to calculate R2 score during training/evaluation
        self.var_targets = torch.var(self.targets)

    def __call__(self, training):
        self.training = training

        r2score = np.zeros(epochs)
        loss_avg = np.zeros(epochs)
        r2score_test = np.zeros(epochs)
        loss_avg_test = np.zeros(epochs)

        for i in range(epochs):
            self.training = True
            run_epoch()
            r2score[i] = self.r2
            loss_avg[i] = self.loss_avg

            self.training = False
            run_epoch()
            r2score_test[i] = self.r2
            loss_avg_test[i] = self.loss_avg

    def run_epoch(self):
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
                batch_predictions, loss = train(features, targets)
            elif not self.training:
                batch_predictions, loss = evaluate(features, targets)

            total_loss += loss
            _predictions.append(batch_predictions)

        self.predictions = torch.cat(_predictions, dim=0)
        self.loss_avg = total_loss.item() / len(self.dataloader)
        self.r2 = 1 - self.criterion(self._predictions,
                                     self.targets) / self.var_targets

    def get_predictions(self):
        return self.predictions

    def get_average_loss(self):
        return self.loss_avg

    def get_r2score(self):
        return self.r2

    def train(self, features, targets):
        predictions = self.model(features)
        loss = self.criterion(predictions, targets)

        self.optimizer.zero_grad()
        # below is ???
        # for param in model.parameters():
        #         param.grad = None
        loss.backward()
        self.optimizer.step()

        return predictions, loss

    def evaluate(self):
        with torch.no_grad():
            predictions = model(features)
            loss = self.criterion(predictions, targets)

        return predictions, loss

    def predict(self, features):
        """Use the trained model to perform predictions on other data.
        """
        with torch.no_grad():
            predictions = model(features)
            loss = self.criterion(predictions, targets)

        return predictions, loss
