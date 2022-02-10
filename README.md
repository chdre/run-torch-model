# run-torch-model
Simple program to run a pytorch compatible model. Includes a tools function which has support for creating dataloader objects required for training/testing/validation.

# Requirements
- torch
- torchmetrics
- scikit-learn

# Install 
Install using pip:
```
pip install run-torch-model
```

# Usage
Use create_dataloader to initiate datasets for training and testing:
```
from run_torch_model import create_dataloader

dataloader_train, dataloader_test = create_dataloader(features=features, 
                                                      targets=targets,
                                                      batch_size=batch_size,
                                                      train_size=train_size,
                                                      test_size=test_size)
```

To run a model we define the optimizer, its arguments and a criterion, feed into the class and perform a call for training.

```
import torch
from run_torch_model import RunTorchNN

optimizer = 'torch.optim.Adam'  # Must be string, if CUDA we initiate the optimizer after calling .cuda for speed-up
optimizer_args = {'lr': 0.001'} # Initialize some arguments for the optimizer
criterion = torch.nn.MSELoss()

run_model = RunTorchNN(model, # Some pytorch model
                          epochs=100, 
                          optimizer=optimizer,
                          optimizer_args=optimizer_args,
                          dataloaders=(dataloader_train, dataloader_test), 
                          criterion=criterion)
 
run_model() # Executes the training
```

To fetch metrics:
```
R2 = run_model.get_r2score()
loss = run_model.get_average_loss()
```

To evaluate the trained model on a different set of features:
```
predictions, loss = run_model.predict(new_features)
```

To evaluate the model on a validation set:
```
loss, r2 = run_model.evaluate(dataloader_validation)
predictions = run_model.get_predictions()  # To get predictions, if necessary 
```
