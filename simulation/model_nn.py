import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class masked_Linear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True,type = 'diag'):
        super(masked_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mask = torch.zeros([out_features, in_features])
        if type == 'diag':
            for i in range(out_features):
                self.mask[i,i] = 1
        elif type == 'jdf':
            for i in range(out_features):
                self.mask[i,i] = 1
                if i%2==0:
                    self.mask[i,i+1] = 1
            self.mask[-1,-3] = 1
        self.mask[:,-2:] = 1
        self.mask = Parameter(self.mask)
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight.masked_fill(self.mask==0, value=0), self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class encoder(nn.Module):
    def __init__(self,struct):
        super(encoder, self).__init__()
        #struct = [n,32,64,K,K+n]
        self.struct = struct
        layers = []
        for i in range(len(self.struct)-1):
            layer = nn.Sequential(
                nn.Linear(self.struct[i],self.struct[i+1]),
                nn.ReLU()
                )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        lifted_x = self.layers(x)
        x = torch.cat((x,lifted_x),-1)
        return x

class decoder(nn.Module):
    def __init__(self,struct):
        super(decoder, self).__init__()
        #struct = [K+n,128,64,32,n]
        self.struct = struct
        layers = []
        for i in range(len(self.struct)-1):
            layer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.struct[i],self.struct[i+1]),
                )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        x = self.layers(x)
        return x

class linear_system(nn.Module):
    def __init__(self,lifeted_state,type='unmasked'):
        super(linear_system, self).__init__()
        if type == 'unmasked':
            self.layer=nn.Linear(lifeted_state+2,lifeted_state,bias=False)
        else:
            self.layer=masked_Linear(lifeted_state+2,lifeted_state,bias=False,type=type)


    def forward(self, x,u):
        x = self.layer(torch.cat((x,u),-1))
        return x

class Koopman(nn.Module):
    def __init__(self,en_struct,de_struct,type='unmasked'):
        super(Koopman, self).__init__()
        
        self.en = encoder(en_struct)
        self.de = decoder(de_struct)
        self.K = linear_system(de_struct[0],type)

    def forward(self,x,u):
        x  = self.en(x)
        x = self.K(x,u)
        prediction = self.de(x)
        return prediction
    
class Loss(nn.Module):
    '''
    Hyperparameter introduction:
    a1 for reconstruction loss
    a2 for prediction loss
    a3 for linearity loss
    a4 for robustness loss
    a5,6 for L2 regulation
    a7 for emphasis on first 3 states in lifted state
    P is the number of steps taken into consideration
    '''
    def __init__(self,a1, a2, a3, a4, a5, a6, a7, P):
        super(Loss, self).__init__()
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.a7 = a7
        self.P = P

    def forward(self, model,x,u):
        # x,u should have 3D structure - (No.trajectory, Time sequence, state)
        '''
        This is used for pytorch 1.10.0 on Google colab
        en = model.get_submodule("en")
        de = model.get_submodule("de")
        K = model.get_submodule("K")
        '''
        # This is used for pytorch 1.4.0 on Yiping's labtop (newer version also works)
        k = 9 # emphasize first two state (position)
        submodules = []
        for idx, m in enumerate(model.children()):
            submodules.append(m)
        en = submodules[0]
        de = submodules[1]
        K = submodules[2]
        #get losses torch.load(saved_model_path)
        # Lx,x = ave(||X(t+i)-decoder(K^i*encoder(Xt))||)
        # Lx,o = ave(||encoder(X(t+i))-K^i*encoder(Xt)||)
        # Lo,x = ave(||Xi-decoder(encoder(Xi))||)
        # Lo,o = ave(||X(t+i)-decoder(K^i*encoder(Xt))||inf)+ave(||X(t+i)-K^i*encoder(Xt)||)
        # Lc,o = first 3 states of lifted state and real state
        mse = nn.MSELoss(reduction='sum')
        Lxx = 0
        Lxo = 0
        Lox = 0
        Loo = 0
        Lco = 0
        K_i_en_x = en(x[:,0,:])
        en_x = en(x)
        de_en_x = de(en_x)
        for i in range(self.P):
            K_i_en_x = K(K_i_en_x,u[:,i,:])
            pred = de(K_i_en_x)
            Lxx += mse(x[:,i+1,:2],pred[:,:2])*k+mse(x[:,i+1,2],pred[:,2])
            Lxo += mse(en_x[:,i+1,:],K_i_en_x)
            #Lco += mse(x[:,i+1,:3],K_i_en_x[:,:3])
            Lox += mse(x[:,i+1,:2],de_en_x[:,i+1,:2])*k+mse(x[:,i+1,2],de_en_x[:,i+1,2])
            Loo += torch.norm(x[:,i+1,:]-pred,p=float("inf"))+torch.norm(x[:,i+1,:]-de_en_x[:,i+1,:],p=float('inf'))
        ave = x.size(0)*self.P
        Lxx /= ave
        Lxo /= ave
        Lox /= ave
        Loo /= ave
        Lco /= ave

        # get regularization
        L2_en = 0
        for param in en.parameters():
            L2_en += (param ** 2).sum()  
        L2_de = 0
        for param in de.parameters():
            L2_de += (param ** 2).sum()  

        # get the sum
        loss = self.a1*Lox + self.a2*Lxx + self.a3*k*Lxo + self.a4*k*Loo + self.a5*L2_en + self.a6*L2_de + self.a7*Lco
        return loss