import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from data_loader import *
from model_nn import *
import csv
import os
import sys
from datetime import datetime
from abc import *
from tqdm import tqdm
import matplotlib.pyplot as plt

class Checkpoint(object):
    
    def __init__(self,arg,filename: str=None):

        self.saved_model_path = filename
        self.arg = arg
        self.num_bad_epochs = 0
        self.best = None

    def early_stopping(self, loss, model):

        if self.best is None:
            self.best = loss
        elif self.best > loss:
            self.best = loss
            self.num_bad_epochs = 0
            self.save(model)
        else:
            self.num_bad_epochs += 1

    def save(self, model):
        """
        Save best models
        arg:
           state: model states
           is_best: boolen flag to indicate whether the model is the best model or not
           saved_model_path: path to save the best model.
        """
        content = {'arg':self.arg,'model':model.state_dict()}
        torch.save(content, self.saved_model_path)
        print("model saved.")

        #torch.save(state, self.saved_model_path)

    def load_saved_model(self, model):
        saved_model_path = self.saved_model_path

        if os.path.isfile(saved_model_path):
            model.load_state_dict(torch.load(saved_model_path)['model'])
        else:
            print("=> no checkpoint found at '{}'".format(saved_model_path))
            
        return model

class CSVLogger():
    def __init__(self, filename, fieldnames=['epoch']):

        self.filename = filename
        self.csv_file = open(filename, 'w')

        # Write model configuration at top of csv
        """
        writer = csv.writer(self.csv_file)
        
        for arg in vars(args):
            writer.writerow([arg, getattr(args, arg)])
        writer.writerow([''])
        """

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()



class Trainer(metaclass=ABCMeta):
    def __init__(self, arg, train_loader, val_loader):    
        # import param
        hyper = arg['hyper']
        en = arg['encoder']
        de = arg['decoder']
        optim = arg['optim']
        type = arg['type']

        # define variables
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        model = Koopman(en,de,type)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)

        self.loss_function = Loss(hyper[0], hyper[1], hyper[2], hyper[3], hyper[4], hyper[5], hyper[6], int(hyper[7]))
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = arg['batch size']

        if optim[0]=='Adadalta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(),lr=optim[1], rho=optim[2])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=optim[1])

    def train(self,  epochs,  checkpoint,  csv_logger):
        # initialize parameters
        self._weight_init(self.model)
        checkpoint.load_saved_model(self.model)
        print("old model loaded.")

        # start training
        patience = min(int(epochs*0.5),100)
        train_loss = []
        val_loss  = []    

        for epoch in range(epochs):
            loss_tra= self.train_one_epoch(epoch)
            loss_tra= self.validate(self.train_loader)
            train_loss.append(loss_tra)
            
            loss_val= self.validate(self.val_loader)
            val_loss.append(loss_val)
            
            tqdm.write('val_loss: %.3f' % (loss_val))
            row = {'epoch': str(epoch+1), 'train_loss': str(loss_tra), 'val_loss': str(loss_val)}
            
            csv_logger.writerow(row)
            checkpoint.early_stopping(loss_val, self.model)
            if checkpoint.num_bad_epochs>=patience:
                tqdm.write("Early stopping with {:.3f} best score, the model did not improve after {} iterations".format(
                        checkpoint.best, checkpoint.num_bad_epochs))
                break
        csv_logger.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_avg = 0.
        progress_bar = tqdm(self.train_loader)
        
        # show progress
        for i, data in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch+1))
            
            X, U=data
            X = X.to(self.device)
            U = U.to(self.device)
            
            self.model.zero_grad()
            loss = self.loss_function(self.model,X,U)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.optimizer.step()
            
            loss_avg +=loss.item()

            progress_bar.set_postfix(loss='%.3f' % (loss_avg / (i + 1)))
        return loss_avg / (i + 1)

    def validate(self, loader):
        self.model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    
        loss_avg = 0.
        
        with torch.no_grad():

            for i, data in enumerate(loader):
                
                X, U=data
                X = X.to(self.device)
                U = U.to(self.device)
                
                loss = self.loss_function(self.model,X,U)      

                loss_avg +=loss.item()
        self.model.train()
        return loss_avg / (i + 1)

    def _weight_init(self, m):    
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data,gain=nn.init.calculate_gain('relu'))
            init.normal_(m.bias.data)


def start_logging(filename):
    f = open('./logs/experiment-{}.txt'.format(filename), 'w')
    sys.stdout = f
    return f

def stop_logging(f):
    f.close()

def train_the_model(train_loader, val_loader, arg, epochs=200):
    # create trainer
    trainer = Trainer(arg, train_loader, val_loader)
    print(f'Trainer created.', end = "\n")

    # define file names
    file_name=f"{arg['type']}_{arg['optim'][0]}_lr_{str(arg['optim'][1])}_hyper_{str(arg['hyper'])}_batch_{str(arg['batch size'])}-{datetime.utcnow().strftime('%m-%d-%H-%M')}"

    csv_logger = CSVLogger(filename=f'./logs/{file_name}.csv',
                       fieldnames=['epoch', 'train_loss', 'val_loss'])
    saved_model_path = './weight/{}.pt'.format(file_name)

    checkpoint = Checkpoint(arg,saved_model_path)

    # initialize recording
    experiment_name = file_name
    stdo = sys.stdout
    f = start_logging(experiment_name)
    print(f'Starting {experiment_name} experiment')
    print(f"Use {arg['type']} matrix as linear system.")
    print(f"Use {arg['optim'][0]} as optimaizer, with learning rate {str(arg['optim'][1])}.")
    print(f"The structure of encoder is {str(arg['encoder'])}, and decoder is {str(arg['decoder'])}.")
    print(f"Hyper parameter used are {str(arg['hyper'])}")
    print(f"The size of batch is {str(arg['batch size'])}")
    # start training
    trainer.train(epochs, checkpoint, csv_logger)
    stop_logging(f)
    sys.stdout = stdo

    return file_name

def plot_learning_curve(file_name):
    # read data
    filename=f'./logs/{file_name}.csv'
    Data = np.loadtxt(open(filename),delimiter=",",skiprows=1)
    epoch = Data[:,0]
    train_loss = Data[:,1]
    val_loss = Data[:,2]
    
    # plot data
    labels = ['train_loss', 'val_loss']
    plt.figure()
    plt.plot(epoch, train_loss, color='r')
    plt.plot(epoch, val_loss, color='k')
    plt.legend(labels=labels)
    plt.show() 

def save_model_as_numpy(file_name):
    # set model
    saved_model_path = './weight/{}.pt'.format(file_name)
    arg = torch.load(saved_model_path)['arg']
    model = Koopman(arg['encoder'],arg['decoder'],arg['type'])
    
    checkpoint = Checkpoint(arg,saved_model_path)
    model=checkpoint.load_saved_model(model)
    
    # create dictionary
    param = {'arg':arg}
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        param[name]=parameters.detach().numpy()
    
    # save as numpy file\
    np.save('./numpy_weight/'+file_name,param)