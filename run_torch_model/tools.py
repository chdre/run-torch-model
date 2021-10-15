import torch
import torch.utils.data as data


def create_dataloader(features, targets, batch_size=128, train_size=0.8, test_size=0.2, validation_size=0.0):
    """Creates a Pytorch compatible dataset of type dataloader. Data is split
    in two or three batches consisting depending on the sizes of train, test
    and validation split.

    :param features: Input (or regressor) for the ML model. Dimensions must be
                     compatible with torch models, i.e. [samples, features] for
                     NN or [samples, channels, features] for a CNN.
    :type features: array_like
    :param targets: self explanatory. Dimensions required to be compatible with
                    torch models, see above.
    :type targets: array_like
    :param batch_size: Size of mini batches
    :type batch_size: int
    :param train_size: Size of training batch
    :type train_size: float
    :param test_size: Size of test batch
    :type test_size: float
    :param validation_size: Size of validation batch
    :type validation_size: float

    :returns data: Tuple of train, test (and validation) dataloaders
    :rtype: tuple of type torch.dataloader
    """

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
