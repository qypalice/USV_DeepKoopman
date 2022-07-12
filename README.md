# USV_DeepKoopman
## Project Description
This repo is for Yiping, Xiangyi and Zilin's capstone project, which designs the path planning and following algorithms for USVs. It mainly includes the modelling, linearisation, simulation and algorithm design for USVs. Thanks a lot for guidance from Ye Wang, who is one of our project supervisors.

## User Guide
### repo structure
All the code are at root directory, including self-defined packages, sample codes and notebook experiment recording. Further explanation is at next section.

./dataset      This folder contains numpy files containing simulated trajectories as datasets\
./logs         This folder contains .txt and .csv files recording learning process\
./rbf_model    This folder contains numpy files saving the parameters of rbf models\
./results      This folder contains .txt files recording the tested loss\
./weight       This folder contains .pt files saving the parameters of NN models

### package introduction
#### data_loader.py
data_loader package includes: (1) functions to generate datasets with different sizes, length, initial states and inputs (1) functions and classes to load datasets and transform into dataloaders that ready for training in Pytorch.

#### model_nn.py
model_nn package includes: (1) Deep Koopman network - encoder, linear system and decoder (2) loss function with given hyperparameters. All the classes inherit from torch.nn.Module.


#### nonlinear_model.py
nonlinear_model provides a function that simulates the nonlinear model of a USV in 3 DOFs, which are surge, sway and yaw.

#### rbf_predictor.py
rbf_predictor package includes: (1) functions for trajectory simulation with lifted states using nonlinear model (2) linear predictor based on given trajectories and different rbf method.

#### train.py
train package includes: (1) checkpoint and csv_logger class to save the model parameters and learning process (2) trainer class to build a trainer for training (3) functions to load dataset, train and test Deep Koopman model, visualization (only for training process, result visualization TODO).

### sample code usage
experiment.py               For Deep Koopman predictor training and testing. \
Koopman_test.ipynb          Same code as experiment.py but run in Jupyter notebook.\
Modelling_rbf_test.ipynb    For linearisation using rbf methods. 
    
