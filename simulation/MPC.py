import sys
import cvxpy as cp
import numpy as np
from numpy import pi
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from nonlinear_model import discrete_nonlinear

# define global parameters
Ts = 0.1
x_min = np.array([-3,-3])
x_max = np.array([3,3])
u_min = np.array([-0.2,-0.3])
u_max = np.array([0.2,0.3])
du_min = np.array([-1.5,-0.5])
du_max = np.array([1.5,0.5])
dU_range = np.array([1.5,0.5])
#eps_min = np.array([-1.,-0.3])
#eps_max = np.array([1.,0.3])

def simulate_path(init_x,SimLength):
    # initialize
    X = np.zeros((2,SimLength+1))
    X[:,0] = init_x.squeeze()[:-1]
    # several step 
    interval = SimLength/6
    for i in range(SimLength):
        if i<interval:# go east
            X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        elif i<2*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,1])
        elif i<3*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,0])#[-1,1])
        #elif i<4*interval:# go west
            #X[:,i+1] = X[:,i]+0.1*np.array([-1,0])
        elif i<5*interval:# go west
            X[:,i+1] = X[:,i]+0.1*np.array([1,-1])#[-1,-1])
        else:
            X[:,i+1] = X[:,i]+0.1*np.array([1,0])#-1])
        '''theta = -(2*(i+1)*pi/SimLength-pi/2)
        X[:,i+1] = np.array([np.cos(theta),np.sin(theta)-1])'''
        #X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        X[:,i+1] = np.maximum(X[:,i+1],x_min)
        X[:,i+1] = np.minimum(X[:,i+1],x_max)
    path = f'./dataset/MPC/SimLenth_{str(SimLength)}_Ts_{str(Ts)}'
    #if not os.path.exists(path):
      #os.makedirs(path)
    sim = {}
    sim['init state'] = init_x
    sim['path'] = X
    np.save(path,sim)
    print(path)

    return path

def get_Augmented_Matrix(A,B,Q,R,rho,Nc):
    """
    input:
    A,B     matrices of linear system
    Q,R     penalty matrices for MPC
    rho     v
    """
    '''determine matrices when input is delta u instead of u
    A_bar = [A      B
            O(m*L)  I(m*m)]
    B_bar = [B
            I(m*m)]
    C = [I(L*L) O(L*m)]
    state = [lifted state
            input u]
    '''
    L = A.shape[0]
    m = B.shape[1]
    A_bar = np.zeros((m+L,m+L))
    A_bar[0:L,0:L] = A
    A_bar[0:L,L:] = B
    A_bar[L:,L:] = np.eye(m)
    B_bar = np.r_[B,np.eye(m)]
    C = np.zeros((L,m+L))
    C[:L,:L] = np.eye(L)

    # get more straight forward matrices for MPC (more steps)
    Gamma = C @ A_bar
    Qbig = np.zeros((L*Nc,L*Nc))
    Qbig[0:L,0:L] = Q
    Theta_r = C @ B_bar
    Theta = np.c_[Theta_r,np.zeros((L,m*(Nc-1)))]
    Rbig = np.zeros((m*Nc,m*Nc))
    Rbig[0:m,0:m] = R
    u_range = np.eye(2)
    for i in range(1,Nc):
        Gamma = np.r_[Gamma,C @ matrix_power(A_bar,i+1)]
        Qbig[i*L:(i+1)*L,i*L:(i+1)*L] = Q
        Rbig[i*m:(i+1)*m,i*m:(i+1)*m] = R
        Theta_r = np.c_[(C @ matrix_power(A_bar,i)) @ B_bar,Theta_r]
        Theta = np.r_[Theta,np.c_[Theta_r,np.zeros((L,m*(Nc-1-i)))]]
        u_range = np.c_[u_range,np.eye(2)]
    # calculate penalty matrix
    rho = rho
    H = Theta.T @ Qbig @ Theta+Rbig
    return Gamma,Theta,Qbig,H,u_range

def MPC_solver_aug(Q,H,Gamma,Theta,rho,Yref,x,u,u_range,Nc):
    # get augmented and reconstructed system relation Y = Gamma*phi+Theta*dU
    phi = np.r_[x,u]
    E = Gamma @ phi - Yref
    G = 2*E.T @ Q @ Theta
    P = E.T @ Q @ E

    # define object function
    dU = cp.Variable(2*Nc)
    eps = cp.Variable(2)
    U = cp.Variable((2*Nc+2))
    obj = cp.Minimize(cp.quad_form(dU, H) + G.T @ dU + P + rho * cp.sum_squares(eps))
    # define constraints
    cons = [eps>=0,
        U[0:2]==u,U[2:]-U[:-2]==dU,u_min@u_range<=U[:-2],u_max@u_range>=U[:-2],
        (du_min-eps)@u_range<=dU,(du_max+eps)@u_range>=dU]
    '''for i in range(Nc):
        cons+=[u_min<=U[2*i+2:2*i+4],u_max>=U[2*i+2:2*i+4],
            du_min-eps<=dU[2*i:2*i+2],du_max+eps>=dU[2*i:2*i+2]]'''
    # define solver
    prob = cp.Problem(obj,cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-6,verbose=False)
    #print(prob.status)
    new_u = U[2:4].value
    if not new_u is None:
        u = new_u
    return u

def MPC_solver(Q,R,rho,A,B,Yref,x,u,Nc,Isplot):
    # define variables -- the combination of dU and slack variable eps
    #group = SX.sym('dU',Np*2)
    L = A.shape[0]
    U = cp.Variable((2,Nc+1))
    Y = cp.Variable((L,Nc+1))
    eps = cp.Variable(2)

    # define object function
    obj = rho * cp.sum_squares(eps)

    # define constraints
    cons = [eps>=0,Y[:,0]==x,U[:,0]==u]
    for t in range(Nc):
        obj += cp.quad_form(Y[:,t+1]-Yref[:,t],Q) + cp.quad_form(U[:,t+1]-U[:,t],R)
        cons += [Y[:,t+1] == A@Y[:,t] + B@U[:,t+1],
                u_min<=U[:,t+1], u_max>=U[:,t+1],
                du_min-eps<=U[:,t+1]-U[:,t], du_max+eps>=U[:,t+1]-U[:,t]]
    # define solver
    prob = cp.Problem(cp.Minimize(obj),cons)
    prob.solve(solver=cp.OSQP, eps_abs=1e-6,verbose=False)
    #print(prob.status)
    #print(eps.value)
    u = U[:,1].value
    # plot the prediction
    if Isplot:
        pred = np.zeros((L,Nc))
        ref = np.zeros((L,Nc))
        for i in range(Nc):
            pred[:,i] = Y[:,i+1].value
            ref[:,i] = Yref[:,i]
        MPC_process_plot(ref,pred,Nc,lifted=True)
    return u


from Koopman_numpy import Koopman_numpy
import time
def MPC_control_process(model_file,ref,init_input,init_state,Q,R,rho,Nc,thre): #temp
    #load model
    operater = Koopman_numpy(model_file)
    A,B = operater.linear_matrix()
    L = A.shape[0]
    Gamma,Theta,Qbig,H,u_range = get_Augmented_Matrix(A,B,Q,R,rho,Nc)

    # generate angle, seperate path into smaller steps
    diff = ref[:,1:]-ref[:,:-1]
    '''angle = np.arctan2(diff[1,:],diff[0,:])
    for i in range(angle.shape[0]-1):
        while angle[i+1]-angle[i]>pi:
            angle[i+1] -= 2*pi
        while angle[i+1]-angle[i]<-pi:
            angle[i+1] += 2*pi'''
    angle = np.arctan(diff[1,:]/diff[0,:])
    for i in range(angle.shape[0]-1):
        while angle[i+1]-angle[i]>pi/2:
            angle[i+1] -= pi
        while angle[i+1]-angle[i]<-pi/2:
            angle[i+1] += pi
    ref = np.r_[ref,np.c_[init_state[2],np.array([angle])]]
    temp = np.zeros((3,0))
    temp = np.c_[temp,ref[:,0]]
    for i in range(1,ref.shape[1]):
        while abs(ref[2,i]-temp[2,-1])>0.03:
            step = np.array([temp[0,-1],temp[1,-1],temp[2,-1]+0.03*np.sign(ref[2,i]-temp[2,-1])])
            temp = np.c_[temp,step]
        if ref[2,i]-temp[2,-1] !=0:
            temp = np.c_[temp,[temp[0,-1],temp[1,-1],ref[2,i]]]
        while abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1])>0.02:
            ratio_x = (ref[0,i]-temp[0,-1])/(abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]))
            ratio_y = (ref[1,i]-temp[1,-1])/(abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]))
            step = np.array([temp[0,-1]+0.02*ratio_x,temp[1,-1]+0.02*ratio_y,temp[2,-1]])
            temp = np.c_[temp,step]
        if abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]) !=0:
            temp = np.c_[temp,ref[:,i]]
    ref = temp
    '''for i in range(ref.shape[1]):
        while ref[2,i]>pi+0.05:
            ref[2,i] -= 2*pi
        while ref[2,i]<-0.05-pi:
            ref[2,i] += 2*pi'''
    for i in range(ref.shape[1]):
        while ref[2,i]>pi+0.05:
            ref[2,i] -= 2*pi
        while ref[2,i]<-pi/2:
            ref[2,i] += pi
        while ref[2,i]<-0.05-pi:
            ref[2,i] += 2*pi
        while ref[2,i]>pi/2:
            ref[2,i] -= pi
    print(ref[2,:])
    
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
    u = init_input
    x = ref[:,0]
    y = lifted_ref[:,0]
    lifted_path = np.zeros((L,0))
    lifted_path = np.c_[lifted_path,y]
    t_avg = 0

    # start contorl simulation
    step = 0
    for i in range(1,ref.shape[1]-Nc):
        j = 0
        while j <30:
            j += 1
            print('Point '+str(i)+' ,Step '+str(j)+' - MSE error in lifted space, ref, state x, input u:')
            '''if x[2]-ref[2,i]>pi:
                x[2] = x[2]-2*pi
                y = operater.encode(x)
            elif x[2]-ref[2,i]<-pi:
                x[2] = x[2]+2*pi
                y = operater.encode(x)'''
            if x[2]-ref[2,i]>pi/2:
                x[2] = x[2]-pi
                y = operater.encode(x)
            elif x[2]-ref[2,i]<-pi/2:
                x[2] = x[2]+pi
                y = operater.encode(x)
            if i < 2: # set parameter
                Isplot = True
            else:
                Isplot = False
            T1 = time.perf_counter() # optimization
            #u = MPC_solver(Q,R,rho,A,B,lifted_ref[:,i:i+Nc],y,u,Nc,Isplot)
            u = MPC_solver_aug(Qbig,H,Gamma,Theta,rho,lifted_ref[:,i:i+Nc].reshape(L*Nc,order='F'),y,u,u_range,Nc)
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
            print(err,ref[:,i],x,u)
            if err<thre:
                break
        #MPC_process_plot(ref_arg,path,path.shape[1],lifted=False)
        step += j
    # plot the lifted space
    #MPC_process_plot(lifted_ref_arg,lifted_path,lifted_path.shape[1],lifted=True)

    # plot
    MPC_process_plot(ref_arg,path,path.shape[1],lifted=False)

    # see the time consumption
    t_avg /= step
    t_avg *= 1000
    print("Average time needed per step is "+str(t_avg)+" ms.")

    # save and see the control result
    file_name = f'Q-{str(np.diag(Q))}_R-{str(np.diag(R))}_rho-{str(rho)}_Nc-{str(Nc)}'
    np.save('./results/MPC/{}'.format(file_name),path)
    err = np.linalg.norm(path-ref_arg)**2 / (path.shape[1])
    print("MSE loss: "+str(err))
    print('Controled path file: '+file_name)
    stdo = sys.stdout
    f = open('./results/MPC/{}.txt'.format(file_name), 'w')
    sys.stdout = f
    print(f'\nMSE loss: {str(err)}.')
    print("Average time needed per step is "+str(t_avg)+" ms.")
    f.close()
    sys.stdout = stdo

    return file_name

def MPC_process_plot(ref,control,N,lifted):
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
        plt.figure(figsize=(4,4))#8,8))
        '''plt.subplot(221)
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

        plt.subplot(223)'''
        plt.plot(ref[0,:N],ref[1,:N],'o-')
        plt.plot(control[0,:N],control[1,:N],'--',linewidth=5)
        plt.grid(True)
        plt.title('position change')
        plt.xlabel('x direction')
        plt.ylabel('y direction')
        plt.legend(legend_list)

        '''plt.subplot(224)
        plt.plot(t,ref[2,:N],'o-')
        plt.plot(t,control[2,:N])
        plt.grid(True)
        plt.xlabel('Time t')
        plt.ylabel('Theta')
        plt.title('Angle change')
        plt.legend(legend_list)'''

        plt.show()
