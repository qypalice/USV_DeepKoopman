# USV_DeepKoopman
## Project Description
This repo is for Yiping, Xiangyi and Zilin's capstone project, which designs the path planning and following algorithms for USVs. It mainly includes the modelling, linearisation, simulation and algorithm design for USVs. Thanks a lot for guidance from Ye Wang, who is one of our project supervisors.

## User Guide
### repo structure
All the code are classified by its usage at root directory,. Further explanation is at next section.
./simulation            This folder contains path following algorithm in simulation\
./ros_implementation    This folder contains algorithm implementation on Aion R6 rover\
./KTMPC                 This ia a temporary folder that contains a new MPC design\

### './simulation' folder introduction
'./simulation' folder is uesd to generate linear model of USV and implement the MPC design. The folder include self-defined packages, sample codes and notebook experiment recording.
#### Subfolders
##### ./dataset
This folder contains numpy files containing simulated trajectories as datasets. The name of the subfolders shows the limitation of the initial state, input and sampling period T_s. In the future, it may also include MPC test trajectories in MPC folder. 'heron_data' includes trajectories from reality, which is currently matlab data needed to be processed.

##### ./logs
This folder contains .txt and .csv files recording learning process.

##### ./results
This folder contains .txt files recording the tested loss. General loss is defined by the loss function during training, see 'model_nn.py' for specific definition. MSE loss is the Mean Squared Error for each step each trajectory.

##### ./weight
This folder contains .pt files saving the parameters of deep neural network models (Deep Koopman Operator). The file name contains the choice of linear system matrix structure, optimizer, hyperparameters and time. In the .pt file, there is a dictionary where 'arg' is the key for its arguments (network structure, batch size, hyperparameter, etc.), and 'model' is the key for the model parameters.

##### ./numpy_weight 
This folder contains .npy files saving the parameters of NN models, transfered from .pt files.

#### Packages
##### data_loader.py
data_loader package includes: (1) functions to generate datasets with different sizes, length, initial states and inputs (2) dataset information visualization (3) functions and classes to load datasets and transform into dataloaders that ready for training in Pytorch.

##### model_nn.py
model_nn package includes: (1) Deep Koopman network - encoder, linear system and decoder (2) loss function with given hyperparameters. All the classes inherit from torch.nn.Module.

##### nonlinear_model.py
nonlinear_model provides a function that simulates the nonlinear model of a USV in 3 DOFs, which are surge, sway and yaw. Input include line velocity and angular velocity.

##### train.py
train package includes: (1) checkpoint and csv_logger class to save the model parameters and learning process (2) trainer class to build a trainer for training (3) functions to load dataset and train Deep Koopman model.

##### test_function.py
test_function provides ways to test the performance of trained models, includes loss calculation and prediction visualization.

##### Koopman_numpy.py
Koopman_numpy defined an class which only relies on numpy package and performs as a Koopman operator, whose parameters are from .npy file.

##### MPC.py
MPC provides functions that (1) generate the testing path for tracking (2) implement Model Predictive Controller parameter (3) simulate to evaluate the performance of the algorithm in efficiency and accuracy.

#### sample code usage
experiment.py               For Deep Koopman Operator training and testing. \
Koopman_test.ipynb          Similar code as experiment.py but run in Jupyter notebook.\
MPC_test.ipynb              For MPC design evaluation and parameter tuning.
