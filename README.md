# run-torch-model
Simple program to run a pytorch compatible model. Currently functions with CNN. Includes a tools function which has support for creating dataloader objects required for training/testing/validation.

# Requirements
- torch

# Install 
Install using pip:
```
pip install git+https://github.com/chdre/run-torch-model/
```

# Usage
Use create_dataloader to initiate datasets for training and testing:
```
from run_torch_model import create_dataloader

dataloader_train, dataloader_test = create_dataloader(features=features, 
                                                      targets=targets,
                                                      batch_size=128,
                                                      train_size=0.8,
                                                      test_size=0.2)
```


```
import torch

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

run_model = RunTorchModel(model, # Some pytorch CNN model
                          epochs=100, 
                          optimizer=optimizer, 
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
