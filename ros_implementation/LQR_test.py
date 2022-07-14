import numpy as np
from Koopman_numpy import Koopman_numpy
import time
from numpy import pi
import matplotlib.pyplot as plt
from nonlinear_model import discrete_nonlinear

# define global parameters
Ts = 0.1
x_min = np.array([-3,-3])
x_max = np.array([3,3])
u_min = np.array([-1.5,-0.5])
u_max = np.array([1.5,0.5])

def simulate_path(init_x,SimLength):
    # initialize
    X = np.zeros((2,SimLength+1))
    X[:,0] = init_x.squeeze()[:-1]
    # several step 
    for i in range(SimLength):
        X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        X[:,i+1] = np.maximum(X[:,i+1],x_min)
        X[:,i+1] = np.minimum(X[:,i+1],x_max)
    path = f'./dataset/SimLenth_{str(SimLength)}_Ts_{str(Ts)}'
    sim = {}
    sim['init state'] = init_x
    sim['path'] = X
    np.save(path,sim)
    print(path)
    return path

def LQR_control_process(model_file,ref,init_state,thre):
    #load model
    operater = Koopman_numpy(model_file)
    A,B = operater.linear_matrix()
    L = A.shape[0]
    K= np.array([[-8.75174299,-8.20800106,0.35489033,9.51706721,1.17981876,-0.05315328, 3.33379727,-1.78509317,0.18732115,-1.09274661,-0.05837545],
                [-5.38380611,5.14141569,2.35592996,0.74023532,4.53593041,-0.024051,0.91312991,-2.58625197,0.20392578,-4.4343588,0.30991247]])
    print(K)

    # generate angle
    diff = ref[:,1:]-ref[:,:-1]
    angle = np.arctan2(diff[1,:],diff[0,:])
    diff = angle[1:]-angle[:-1]
    #print(ref[1,:])
    print(angle)
    ref = np.r_[ref,np.c_[init_state[2],np.array([angle])]]
    temp = np.zeros((3,0))
    temp = np.c_[temp,ref[:,0]]
    for i in range(1,ref.shape[1]):
        while abs(ref[2,i]-temp[2,-1])>0.03:
            step = np.array([temp[0,-1],temp[1,-1],temp[2,-1]+0.05*np.sign(ref[2,i]-temp[2,-1])])
            temp = np.c_[temp,step]
        if ref[2,i]-temp[2,-1] !=0:
            temp = np.c_[temp,[temp[0,-1],temp[1,-1],ref[2,i]]]
        while abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1])>0.1:
            ratio_x = (ref[0,i]-temp[0,-1])/(abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]))
            ratio_y = (ref[1,i]-temp[1,-1])/(abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]))
            step = np.array([temp[0,-1]+0.2*ratio_x,temp[1,-1]+0.2*ratio_y,temp[2,-1]])
            temp = np.c_[temp,step]
        if abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]) !=0:
            temp = np.c_[temp,ref[:,i]]
    ref = temp
    
    # lift the reference
    lifted_ref = np.zeros((L,ref.shape[1]))
    for i in range(ref.shape[1]):
        lifted_ref[:,i] = operater.encode(ref[:,i])

    lifted_ref_arg = np.zeros((L,0))
    ref_arg = np.zeros((3,0))
    lifted_ref_arg = np.c_[lifted_ref_arg,lifted_ref[:,0]]
    ref_arg = np.c_[ref_arg,ref[:,0]]
    
    # initialization
    path = np.zeros((3,0))
    path = np.c_[path,ref[:,0]]
    x = ref[:,0]
    y = lifted_ref[:,0]
    lifted_path = np.zeros((L,0))
    lifted_path = np.c_[lifted_path,y]
    t_avg = 0

    # start contorl simulation
    step = 0
    for i in range(1,ref.shape[1]):
        j = 0
        while j <30:
            j += 1
            print('Point '+str(i)+' ,Step '+str(j)+' - MSE error in lifted space,state x, input u:')
            if x[2]>pi:
                x[2] = x[2]-2*pi
                y = operater.encode(x)
            elif x[2]<-pi:
                x[2] = x[2]+2*pi
                y = operater.encode(x)
            T1 = time.perf_counter() # optimization
            u = K@(y-lifted_ref[:,i])
            u = np.maximum(u,u_min)
            u = np.minimum(u,u_max)
            T2 = time.perf_counter()
            t_avg += T2-T1
            # record for each step
            x = discrete_nonlinear(x,u,Ts).squeeze()
            y = operater.encode(x)
            print(ref[:,i])
            ref_arg = np.c_[ref_arg,ref[:,i]]
            path = np.c_[path,x]
            lifted_ref_arg = np.c_[lifted_ref_arg,lifted_ref[:,i]]
            lifted_path = np.c_[lifted_path,y]
            err = np.linalg.norm(x-ref[:,i])
            print(err,x,u)
            if err<thre:
                break
        #LQR_process_plot(ref_arg,path,path.shape[1],lifted=False)
        step += j
    # plot the lifted space
    LQR_process_plot(lifted_ref_arg,lifted_path,lifted_path.shape[1],lifted=True)

    # plot
    LQR_process_plot(ref_arg,path,path.shape[1],lifted=False)

    # see the time consumption
    t_avg /= step
    t_avg *= 1000
    print("Average time needed per step is "+str(t_avg)+" ms.")

    # save and see the control result
    err = np.linalg.norm(path-ref_arg)**2 / (path.shape[1])
    print("MSE loss: "+str(err))

def LQR_process_plot(ref,control,N,lifted):
    t = np.linspace(1,N,N)
    legend_list = ['ref','control']

    # plot
    if lifted:
        k = int(ref.shape[0]/3)+1
        plt.figure(figsize=(16,48/k))
        for i in range(ref.shape[0]):
            plt.subplot(3,k,i+1)
            plt.plot(t,ref[i,:N],'o-')
            plt.plot(t,control[i,:N])
            plt.grid(True)
            plt.xlabel('Time t')
            plt.legend(legend_list)
        plt.show()
    else:
        plt.figure(figsize=(8,8))
        plt.subplot(221)
        plt.plot(t,ref[0,:N],'o-')
        plt.plot(t,control[0,:N])
        plt.grid(True)
        plt.xlabel('Time t')
        plt.ylabel('x direction')
        plt.title('X position change')
        plt.legend(legend_list)

        plt.subplot(222)
        plt.plot(t,ref[1,:N],'o-')
        plt.plot(t,control[1,:N])
        plt.grid(True)
        plt.title('Y position change')
        plt.xlabel('Time t')
        plt.ylabel('y direction')
        plt.legend(legend_list)

        plt.subplot(223)
        plt.plot(ref[0,:N],ref[1,:N],'o-')
        plt.plot(control[0,:N],control[1,:N])
        plt.grid(True)
        plt.title('position change')
        plt.xlabel('x direction')
        plt.ylabel('y direction')
        plt.legend(legend_list)

        plt.subplot(224)
        plt.plot(t,ref[2,:N],'o-')
        plt.plot(t,control[2,:N])
        plt.grid(True)
        plt.xlabel('Time t')
        plt.ylabel('Theta')
        plt.title('Angle change')
        plt.legend(legend_list)

        plt.show()

if __name__ == '__main__':
    SimLength = 10
    init_x = np.array([0.,0.,0.])
    path=simulate_path(init_x,SimLength)
    model_file = 'unmasked_encoder_[3, 32, 64, 8]_decoder_[11, 128, 64, 32, 3]_hyper_[1.0, 3.0, 0.3, 1e-07, 1e-07, 1e-07, 10]_batch_10'
    ref = np.load(path+".npy",allow_pickle=True).item()
    path_ref = ref['path']
    x_init = ref['init state']
    thre = 10
    LQR_control_process(model_file,path_ref,x_init,thre)

