import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from train import Checkpoint
from model_nn import Koopman,Loss
import sys


def test_the_model(test_loader, file_name):
    # set model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #get arguments
    saved_model_path = './weight/{}.pt'.format(file_name)
    arg = torch.load(saved_model_path)['arg']    
    #loss function
    hyper = arg['hyper']
    loss_function = Loss(hyper[0], hyper[1], hyper[2], hyper[3], hyper[4], hyper[5], hyper[6], int(hyper[7]))
    mse = torch.nn.MSELoss()
    #load model
    model = Koopman(arg['encoder'],arg['decoder'],arg['type'])
    checkpoint = Checkpoint(arg,saved_model_path)
    model=checkpoint.load_saved_model(model)
    model = model.to(device)
    model.eval()
    
    #start calculation
    loss_avg = 0.
    score_avg = 0.
    progress_bar = tqdm(test_loader)
    with torch.no_grad():
        for i, data in enumerate(progress_bar):
            X, U=data
            X = X.to(device)
            U = U.to(device)
            loss = loss_function(model,X,U)
            Y = get_prediction(X,U,model)
            score = mse(X,Y)
            loss_avg +=loss.item()
            score_avg += score.item()
    loss_avg = loss_avg / (i + 1)
    score_avg = score_avg / (i + 1)

    #print result in kernal and txt file
    stdo = sys.stdout
    print(f'\nGeneral loss: {str(loss_avg)}.')
    print(f'\nMSE loss: {str(score_avg)}.')
    f = open('./results/{}.txt'.format(file_name), 'w')
    sys.stdout = f
    print(f'\nGeneral loss: {str(loss_avg)}.')
    print(f'\nMSE loss: {str(score_avg)}.')
    f.close()
    sys.stdout = stdo

def get_prediction(X,U, model):
    # get encoder,linear system, decoder
    submodules = []
    for idx, m in enumerate(model.children()):
        submodules.append(m)
    en = submodules[0]
    de = submodules[1]
    K = submodules[2]

    Y = X.clone()
    K_i_en_x = en(X[:,0,:])
    for i in range(1,X.shape[1]):
        K_i_en_x = K(K_i_en_x,U[:,i-1,:].clone())
        Y[:,i,:] = de(K_i_en_x)
    return Y

def position_plot(preds,truth):
    # get data and set parameters
    preds = preds
    truth = truth
    num_model = preds.shape[0]
    N = truth.shape[0]
    t = np.linspace(1,N,N)
    legend_list = ['non_linear']
    for i in range(num_model):
        legend_list.append('model '+str(i+1))

    # plot
    plt.figure(figsize=(9,9))
    plt.subplot(221)
    plt.plot(t,truth[:,0],'o-')
    for i in range(num_model):
        plt.plot(t,preds[i,:,0])
    plt.grid(True)
    plt.xlabel('Time t')
    plt.ylabel('x direction')
    plt.title('X position change')
    plt.legend(legend_list)

    plt.subplot(222)
    plt.plot(t,truth[:,1],'o-')
    for i in range(num_model):
        plt.plot(t,preds[i,:,1])
    plt.grid(True)
    plt.grid(True)
    plt.title('Y position change')
    plt.xlabel('Time t')
    plt.ylabel('y direction')
    plt.legend(legend_list)
    
    plt.subplot(223)
    plt.plot(truth[:,0],truth[:,1],'o-')
    for i in range(num_model):
        plt.plot(preds[i,:,0],preds[i,:,1])
    plt.grid(True)
    plt.grid(True)
    plt.title('Angle change')
    plt.xlabel('x direction')
    plt.ylabel('y direction')
    plt.legend(legend_list)

    plt.subplot(224)
    plt.plot(t,truth[:,2],'o-')
    for i in range(num_model):
        plt.plot(t,preds[i,:,2])
    plt.grid(True)
    plt.grid(True)
    plt.title('Angle change')
    plt.xlabel('Time t')
    plt.ylabel('$\psi$ (rad)')
    plt.legend(legend_list)
    plt.show()

def result_sample(data_path,file_names,index=0):
    # get data
    xx = np.load(data_path+"/X_test.npy")
    uu = np.load(data_path+"/U_test.npy")
    #xx = np.load(data_path+"/X_train.npy")
    #uu = np.load(data_path+"/U_train.npy")
    xx = xx[index]
    uu = uu[index]
    yy = np.empty((len(file_names),xx.shape[0],xx.shape[1]))
    #uu = np.array([[0.5,0.5,0.5]])
    #U = torch.tensor(uu).float() 
    X = torch.tensor(np.atleast_2d(xx)).float().unsqueeze(0)#xx[0])).float()
    U = torch.tensor(np.atleast_2d(uu)).float().unsqueeze(0)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    U = U.to(device)
    
    i = 0
    for file_name in file_names:
        #load model
        saved_model_path = './weight/{}.pt'.format(file_name)
        arg = torch.load(saved_model_path)['arg']    
        model = Koopman(arg['encoder'],arg['decoder'],arg['type'])
        checkpoint = Checkpoint(arg,saved_model_path)
        model=checkpoint.load_saved_model(model)
        model = model.to(device)
        model.eval()
        
        # get prediction
        Y = get_prediction(X,U, model)
        yy[i,:,:] = Y.cpu().detach().numpy().squeeze()
        i += 1
    position_plot(yy,xx)
