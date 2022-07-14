import torch
import numpy as np
from tqdm import tqdm
from numpy.random import rand,randint,uniform
from nonlinear_model import discrete_nonlinear
import matplotlib.pyplot as plt
import os
from numpy import pi

du_range = np.array([1,0.5])
def simulation(x_range,u_range,SimLength=10,Ntraj=1000,Ts=0.01):
    '''
    This function is to get simulated trajectories of the USV, and save 
    the trajectories and relevant input in numpy file. 

    Input Parameters:
    noise_range     a small float to show the noise range of the initial state
    dest_range      a large 2*1 numpy array, to show the range of the destination
    K               The proportional feedback controller parameter
    SimLength       the length of the trajectory, unit is the sampling period
    Ntraj           the number of trajectories, i.e. size of the dataset
    Ts              sampling period

    Return:
    filename is to provide the parameter information of the dataset

    Simulation rules:
    (i)  For random input in [-u_range,u_range]
    (ii) For initial state, it is fixed with noise added.(Vary if needed, TBA)
    '''
    # generate x
    X = np.empty((Ntraj,SimLength+1,3))
    # generate u
    '''
    Rule 1: consider the symmetry in input
    U0 = uniform(low=-u_range, high=u_range, size=(int(Ntraj/4),SimLength,2))
    for i in range(SimLength):
        U0[:,i,:] = U0[:,i-1,:]+uniform(low=-du_range, high=du_range, size=(int(Ntraj/4),2))
        for j in range(int(Ntraj/4)):
            U0[j,i,:] = np.fmin(U0[j,i,:],u_range)
            U0[j,i,:] = np.fmax(U0[j,i,:],-u_range)
    U_v = U0[:,:,0]
    U_w = U0[:,:,1]
    U1 = np.stack((-U_v,U_w),axis=2)
    U2 = np.stack((U_v,-U_w),axis=2)
    U3 = np.stack((-U_v,-U_w),axis=2)
    U = np.vstack((U0,U1,U2,U3))'''
    U = uniform(low=-u_range, high=u_range, size=(Ntraj,SimLength,2))
    '''
    Rule 2: consider the limited input change
    for i in range(SimLength):
        U[:,i,:] = U[:,i-1,:]+uniform(low=-du_range, high=du_range, size=(Ntraj,2))
        for j in range(int(Ntraj/4)):
            U[j,i,:] = np.fmin(U[j,i,:],u_range)
            U[j,i,:] = np.fmax(U[j,i,:],-u_range)
    '''
    
    pbar = tqdm(total=Ntraj)
    for i in range(Ntraj):
        xx = np.empty((SimLength+1,3))
        # Intial state is a random vector within given range
        x = uniform(low=-x_range, high=x_range, size=(1,3))+rand(1,3)*0.1
        x = x.squeeze()
        xx[0,:] = x

        # Simulate one trajectory
        for j in range(SimLength):
            u = U[i,j,:]
            x = discrete_nonlinear(x,u,Ts)
            x = x.squeeze()
            xx[j+1,:] = x
        # Store
        X[i,:,:] = xx
        pbar.update(1)
    pbar.close()
    return X,U

def produce_dataset(x_range,u_range,SimLength=10,Ntraj=1000,Ts=0.01):
    print("Start simulating...")
    X_train,U_train = simulation(x_range,u_range,SimLength,int(0.6*Ntraj),Ts)
    X_val,U_val = simulation(x_range,u_range,SimLength,int(0.2*Ntraj),Ts)
    X_test,U_test = simulation(x_range,u_range,SimLength,int(0.2*Ntraj),Ts)

    # save the matrix
    print("\nDataset produced.")
    path = f'./dataset/state-{str(x_range)}_input-{str(u_range)}_{str(SimLength)}x{str(Ntraj)}_Ts_{str(Ts)}'
    if not os.path.exists(path):
      os.makedirs(path)
    np.save(path+"/X_train",X_train)
    np.save(path+"/U_train",U_train)
    np.save(path+"/X_val",X_val)
    np.save(path+"/U_val",U_val)
    np.save(path+"/X_test",X_test)
    np.save(path+"/U_test",U_test)
    print("Dataset saved.")
    return path

def dataset_plot(data_path,num=100):
    x = np.load(data_path+"/X_train.npy")
    u = np.load(data_path+"/U_train.npy")
    x = x.squeeze()
    N = x.shape[1]
    t = np.linspace(1,N,N)

    plt.figure(figsize=(9,9))
    plt.subplot(221)
    for i in range(num):
        plt.plot(x[i,:,0],x[i,:,1], 'o-')
    plt.grid(True)
    plt.xlabel('x direction')
    plt.ylabel('y direction')
    plt.title('location change')

    plt.subplot(222)
    for i in range(num):
        plt.plot(t,x[i,:,2])
    plt.grid(True)
    plt.title('Angle change')
    plt.xlabel('Time t')
    plt.ylabel('$\psi$ (rad)')

    plt.subplot(223)
    for i in range(num):
        plt.plot(t[:-1],u[i,:,0])
    plt.grid(True)
    plt.xlabel('Time t')
    plt.ylabel('v m/s')
    plt.title('Forward speed')

    plt.subplot(224)
    for i in range(num):
        plt.plot(t[:-1],u[i,:,1])
    plt.grid(True)
    plt.title('w m/s')
    plt.xlabel('Time t')
    plt.ylabel('Rotation speed')
    plt.show()


def get_data(path):
    xx_train = np.load(path+"/X_train.npy")
    xx_val = np.load(path+"/X_val.npy")
    xx_test = np.load(path+"/X_test.npy")
    uu_train = np.load(path+"/U_train.npy")
    uu_val = np.load(path+"/U_val.npy")
    uu_test = np.load(path+"/U_test.npy")
    X_train = torch.tensor(xx_train).float()
    U_train = torch.tensor(uu_train).float()
    X_val = torch.tensor(xx_val).float()
    U_val = torch.tensor(uu_val).float()
    X_test = torch.tensor(xx_test).float()
    U_test = torch.tensor(uu_test).float()
    return  X_train,U_train,X_val,U_val,X_test,U_test      

#==========================================================================
# Dataset loader
#==========================================================================
class Dataset(torch.utils.data.Dataset):
    def __init__(self,X,U):
        self.X  = X
        self.U  = U
        
    def __len__(self):
        return self.U.shape[0]

    def __getitem__(self, index):
        xx = self.X[index]
        uu = self.U[index]
        return xx,uu
        
        
def get_loaders(X,U,batch_size=10):
    data = Dataset(X,U)
    loader=torch.utils.data.DataLoader(data, batch_size, shuffle=True, num_workers=2,drop_last=False)
    return loader

def get_dataloaders(path,batch_size=10):
    X_train,U_train,X_val,U_val,X_test,U_test = get_data(path)

    # get dataloaders
    tra_loader = get_loaders(X_train,U_train,batch_size)
    val_loader = get_loaders(X_val,U_val,batch_size)
    test_loader = get_loaders(X_test,U_test,batch_size)
    return tra_loader, val_loader, test_loader