import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchmetrics.functional import r2_score


class RunTorchNN:
    """Simple program to execute training for a pytorch compatible NN.

    The training will be performed on CUDA if this is availible, else it will
    run on CPU.

    The program can be used to perform predictions on new samples by passing
    dataloaders as None. In this case several functions will not be possible
    to execute, but using predict() and other side functions will work. At the
    moment it is not specified which functions requires a dataloader, but we
    expect that the average user, which is familiar with neural networks, is
    able to understand what are the requirements when using the package outside
    the main purpose.

    :param model: Pytorch model
    :type model: torch.nn.Module
    :param epochs: Number of epochs to train, i.e. number of training iterations
    :type epochs: int
    :param optimizer: String telling which tytorch optimizer to use, e.g.
                      'torch.optim.Adam'.
    :type optimizer: str
    :param optimizer_args: Parameters for chosen optimizer.
    :type args: dict
    :param dataloaders: Datasets containing features and targets
    :type dataloaders: tuple of type torch.data.dataloader
    :param criterion: Loss function for NN
    :type criterion: torch.nn.module.loss
    :param verbose: Set training to print running loss and R2 to terminal.
                    Defaults to False. Will significantly slow down the training
                    if using a GPU.
    :type verbose: bool
    :param seed: Seed for torch. Defaults to 42.
    :type seed: int
    """

    def __init__(self, model, epochs, optimizer, optimizer_args, dataloaders,
                 criterion, verbose=False, seed=42):
        self.model = model
        self.epochs = epochs

        # Automatically set cuda as device if it is available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = eval(optimizer)(
            self.model.parameters(), **optimizer_args)
        self.criterion = criterion.to(self.device)
        self.verbose = verbose

        if dataloaders is not None:
            if len(dataloaders) >= 2:
                self.dataloader_train = dataloaders[0]
                self.dataloader_test = dataloaders[1]
                if len(dataloaders) == 3:
                    self.dataloader_valid = dataloaders[2]

        torch.manual_seed(seed)

    def __call__(self):
        """Calling the class results in training and testing for a specified
        number of epochs, i.e. training iterations.
        """
        self.r2score_train = torch.zeros(
            self.epochs, device=torch.device(self.device))
        self.loss_avg_train = torch.zeros(
            self.epochs, device=torch.device(self.device))
        self.r2score_test = torch.zeros(
            self.epochs, device=torch.device(self.device))
        self.loss_avg_test = torch.zeros(
            self.epochs, device=torch.device(self.device))

        for i in range(self.epochs):
            self.training = True
            self.run_epoch(self.dataloader_train)
            self.r2score_train[i] = self.r2
            self.loss_avg_train[i] = self.loss_avg

            self.training = False
            self.run_epoch(self.dataloader_test)
            self.r2score_test[i] = self.r2
            self.loss_avg_test[i] = self.loss_avg

            if self.verbose:
                self.verbose_call(i)
            # WIP
            # if self.intermediate_saving and i == self.save_interval:
            #     self.save_model(location=f'{self.save_path}/model_e{i}.pt')

    def run_epoch(self, dataloader):
        """Executes a single training iteration.

        :param dataloader: Data containing features and targets.
        :type dataloader: torch.data.DataLoader
        """
        # If training is False model goes to eval
        self.model.train(self.training)

        _predictions = []
        # In the case where device is GPU it is faster to work on GPU with Tensor than float on CPU
        total_loss = torch.zeros(1, device=torch.device(self.device))

        # Dummy tensor to use for concatenating all tensors. Faster than list
        all_targets = torch.Tensor([]).to(self.device)

        for data_batch in dataloader:
            features = data_batch[0].to(self.device)
            targets = data_batch[1].to(self.device)
            all_targets = torch.cat((all_targets, targets), dim=0)

            if self.training:
                self.train(features, targets)
            else:
                self.mtest(features, targets)

            total_loss += self.batch_loss
            _predictions.append(self.batch_predictions)

        self.predictions = torch.cat(_predictions, dim=0)
        self.loss_avg = total_loss.item() / len(dataloader)

        assert self.predictions.shape == all_targets.shape, \
            'self.predictions.shape == all_targets.shape'
        self.r2 = r2_score(preds=self.predictions, target=all_targets)

    def train(self, features, targets):
        """Train run for model.

        :param features: self explanatory
        :type features: torch.Tensor
        :param targets: self explanatory
        :type targets: torch.Tensor
        """
        self.batch_predictions = self.model(features)
        self.batch_loss = self.criterion(self.batch_predictions, targets)

        # self.optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

        self.batch_loss.backward()
        self.optimizer.step()

    def mtest(self, features, targets):
        """Test run for model.

        :param features: self explanatory
        :type features: torch.Tensor
        :param targets: self explanatory
        :type targets: torch.Tensor
        """
        with torch.no_grad():
            self.batch_predictions = self.model(features)
            self.batch_loss = self.criterion(self.batch_predictions, targets)

    def verbose_call(self, i):
        """Function will be called if verbose is set to True. This will slow
        down the training significantly if training is done on GPU. The reason
        for this is casting from GPU to CPU is slow.

        TO-DO:
            - Make this function a static method

        :param i: Training iteration.
        :type i: int
        """
        print(
            f'Epoch: {i + 1}/{self.epochs}  |  [TRAIN,TEST] -- loss: [{self.loss_avg_train[i].item():.2f}, {self.loss_avg_test[i].item():.2f}]   |  R2: [{self.r2score_train[i].item():.2f}, {self.r2score_test[i].item():.2f}]', flush=True)

    def get_predictions(self):
        """Returns predictions. Function could return predictions from
        training, testing or validation, show care when using. Will most likely
        be removed as it is kind of useless.
        """
        return self.predictions

    def get_growing_loss(self):
        """Returns train and test loss as a function of epochs."""
        return self.loss_avg_train.detach().numpy(), self.loss_avg_test.detach().numpy()

    def get_growing_r2(self):
        """Returns train and test R2 score as a function of epochs."""
        return self.r2score_train.detach().numpy(), self.r2score_test.detach().numpy()

    def get_average_loss(self):
        """Returns average loss."""
        return self.loss_avg

    def get_r2score(self):
        """Returns R2 score."""
        return self.r2.item()

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
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)

        r2 = r2_score(self.predictions, all_targets)

        return loss, r2

    def predict(self, features):
        """Use the trained model to perform predictions on unseen data.

        :param features: A set of features which are compatible with the model
        :type features: torch.Tensor
        :returns predictions: self explanatory
        :rtype predictions: torch.Tensor
        """

        features = features.to(self.device)

        with torch.no_grad():
            predictions = self.model(features)

        return predictions

    @staticmethod
    def numpy_save(location, tensor):
        """Saves a given tensor as an array at a specified location. Arrays are
        saved with numpy.save.

        :param location: Path, including filename, to store the tensor.
        :type location: Pathlib.PosixPath or str
        :param tensor: Tensor to store as numpy array
        :type tensor: torch.Tensor
        """
        if tensor.is_cuda:
            np.save(location, tensor.detach().cpu().numpy())
        else:
            np.save(location, tensor.detach().numpy())

    @staticmethod
    def iterate_filename(location):
        """If file exists we create a new filename instead of overwriting. This
        function adds a number to the chosen filename.

        :param location: Path to store the file, including the filename.
        :type location: Pathlib.PosixPath
        :returns filename: New path with added integer at the end of the
                           filename.
        :rtype filename: Pathlib.PosixPath
        """
        i = 1
        while True:
            filename = Path(str(location) + f'_{i}')
            if Path(str(filename) + '.npy').is_file():
                i += 1
            else:
                break

        return filename

    def save_running_metrics(self, location, suffix='', overwrite=False):
        """Saves running metrics to file. Files are stored using numpy.save.
        Tensors are broadcasted cpu in case of cuda.

        :param location: Where to save the metric.
        :type location: str
        :param suffix: Suffix for filename
        :type suffix: str
        :param overwrite: Overwrite if file exits.
        :type overwrite: bool
        """
        if not isinstance(location, str):
            raise ValueError('Argument location must be of type str')

        path = Path(location)

        if overwrite:
            self.numpy_save(
                path / f'r2_train{suffix}', self.r2score_train)
            self.numpy_save(
                path / f'r2_test{suffix}', self.r2score_test)
            self.numpy_save(
                path / f'loss_avg_train{suffix}', self.loss_avg_train)
            self.numpy_save(
                path / f'loss_avg_test{suffix}', self.loss_avg_test)

        else:
            if (path / f'r2_train{suffix}.npy').is_file():
                p = self.iterate_filename(path / f'r2_train{suffix}')
                self.numpy_save(p, self.r2score_train)
            else:
                self.numpy_save(
                    path / f'r2_train{suffix}', self.r2score_train)

            if (path / f'r2_test{suffix}.npy').is_file():
                p = self.iterate_filename(path / f'r2_test{suffix}')
                self.numpy_save(p, self.r2score_test)
            else:
                self.numpy_save(
                    path / f'r2_test{suffix}', self.r2score_test)

            if (path / f'loss_avg_train{suffix}.npy').is_file():
                p = self.iterate_filename(path / f'loss_avg_train{suffix}')
                self.numpy_save(p, self.loss_avg_train)
            else:
                self.numpy_save(
                    path / f'loss_avg_train{suffix}', self.loss_avg_train)

            if (path / f'loss_avg_test{suffix}.npy').is_file():
                p = self.iterate_filename(path / f'loss_avg_test{suffix}')
                self.numpy_save(p, self.loss_avg_test)
            else:
                self.numpy_save(
                    path / f'loss_avg_test{suffix}', self.loss_avg_test)

    def save_metric(self, location, metric):
        """Saves specific metric to file.

        Requires knowledge of parameters in model to be useful?

        :param location: Where to save the metric.
        :type location: str
        :param metric: Which metric to save, typically R2/MSE.
        :type metric: str
        """
        raise NotImplementedError('To be added if necessary')

    def save_model(self, location):
        """Saves trained model to location.

        :param location: Where to save the model and the name of the file.
        :type location: str
        """
        torch.save({'state_dict': self.model.state_dict()}, location)

    def load_model(self, location, eval=True):
        """Loads a saved model.

        :param location: Where to load the model and the name of the file.
        :type location: str
        """
        checkpoint = torch.load(
            location, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        if eval:
            self.model.eval()
        else:
            self.model.train()

    def intermediate_saving(self, save, interval, path):
        """Turns on or off intermediate saving during training for a fixed
        interval.

        :param save: Whether to activate saving or not.
        :type save: bool
        :param interval: How often to save the model.
        :type interval: int
        """
        self.intermediate_saving = save
        self.save_interval = interval
        self.save_path = path
