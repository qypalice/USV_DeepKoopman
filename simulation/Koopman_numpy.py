import numpy as np

def relu(x):
    return x * (x > 0)

class Koopman_numpy:
    # initialise the neural network
    def __init__(self, file_name):
        params = np.load('./numpy_weight/'+file_name+'.npy',allow_pickle=True).item()
        self.en = {'weight':[],'bias':[]}
        self.de = {'weight':[],'bias':[]}
        for name,param in params.items():
            if name[:2]=='en':
                if name[-6:]=='weight':
                    self.en['weight'].append(param)
                else:
                    self.en['bias'].append(param)
            elif name[:2]=='de':
                if name[-6:]=='weight':
                    self.de['weight'].append(param)
                else:
                    self.de['bias'].append(param)
        if 'K.layer.mask' in params:
            K = params['K.layer.weight']*params['K.layer.mask']
        else:
            K = params['K.layer.weight']
        self.A = K[:,:-2]
        self.B = K[:,-2:]
        
        
    def encode(self,x):
        lifted_x = x
        for i in range(len(self.en['weight'])):
            lifted_x = self.en['weight'][i]@lifted_x+self.en['bias'][i]
            lifted_x = relu(lifted_x)
        x = np.r_[x,lifted_x]
        return x

    def decode(self,x):
        for i in range(len(self.de['weight'])):
            x = relu(x)
            x = self.de['weight'][i]@x+self.de['bias'][i]
        return x

    def linear(self,x,u):
        x = self.A @ x + self.B@u
        return x

    def linear_matrix(self):
        return self.A,self.B

    def property(self):
        controllability_matrix = self.B
        controllability_column = self.B
        for _ in range(self.A.shape[0]-1):
            controllability_column = self.A @ controllability_column
            controllability_matrix = np.c_[controllability_matrix,controllability_column]
        if np.linalg.matrix_rank(controllability_matrix) == self.A.shape[0]:
            print("The system is controllable.")
        else:
            print("The system is uncontrollable, the rank of the controllability matrix is "+str(np.linalg.matrix_rank(controllability_matrix))+".")





