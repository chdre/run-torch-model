import torch
import torch.utils.data as data
import numpy as np
import warnings


def create_dataloader(features, targets, batch_size, train_size=0.8, test_size=0.2, validation_size=0, seed=42):
    """Creates a Pytorch compatible dataset of type dataloader. Data is split
    in two or three batches consisting depending on the sizes of train, test
    and validation split.

    :param features: Input (or regressor) or location of file containing the
                     features for the ML model. If location is supplied the file
                     must be compatible with numpy.load. Dimensions must be
                     compatible with torch models, i.e. [samples, features] for
                     NN or [samples, channels, features] for a CNN.
    :type features: array_like or str
    :param targets: Targets or or location of file containing the targets. If
                    location is supplied the file must be compatible with
                    numpy.load. Dimensions required to be compatible with torch
                    models, see above.
    :type targets: array_like or str
    :param batch_size: Size of mini batches.
    :type batch_size: int or
    :param train_size: Size of training batch. Defaults to 0.8.
    :type train_size: int or float
    :param test_size: Size of test batch. Defaults to 0.2.
    :type test_size: int or float
    :param validation_size: Size of validation batch. Defaults to 0.
    :type validation_size: int or float
    :param seed: Seed for torch random split. Defaults to 42.
    :type seed: int

    :returns data: Tuple of train, test (and validation) dataloaders
    :rtype: tuple of type torch.dataloader
    """
    torch.manual_seed(seed)

    if isinstance(features, str):
        features = np.load(features)
        if len(features.shape) == 3:
            features = features[:, np.newaxis, :, :]
    if isinstance(targets, str):
        targets = np.load(targets)
        if len(targets.shape) == 1:
            targets = targets[:, np.newaxis]

    nf = features.shape[0]
    nt = targets.shape[0]
    assert nf == nt, 'Number of samples for targets and features does not match'
    if isinstance(train_size, float):
        train_size = int(nf * train_size)
    if isinstance(test_size, float):
        test_size = int(nf * test_size)
    if isinstance(validation_size, float):
        validation_size = int(nf * validation_size)

    if not train_size + test_size + validation_size == nf:
        train_size += nf - (train_size + test_size + validation_size)
        warnings.warn(
            'Train, test and validation size does not add to length of dataset. Added rest of samples to training set.')

    x = torch.Tensor(features)
    y = torch.Tensor(targets)

    dataset = data.TensorDataset(x, y)

    train, test, validation = data.random_split(
        dataset, (train_size, test_size, validation_size))

    dataloader_train = data.DataLoader(train, batch_size=batch_size)
    dataloader_test = data.DataLoader(test, batch_size=batch_size)
    dataloader_valid = data.DataLoader(validation, batch_size=batch_size)

    if validation_size != 0:
        return_data = (dataloader_train, dataloader_test, dataloader_valid)
    else:
        return_data = (dataloader_train, dataloader_test)

    return return_data
