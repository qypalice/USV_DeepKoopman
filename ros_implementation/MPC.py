import cvxpy as cp
import numpy as np
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt

# define global parameters
Ts = 0.1
x_min = np.array([-3,-3])
x_max = np.array([3,3])
u_min = np.array([-0.3,-0.3])
u_max = np.array([0.3,0.3])
du_min = np.array([-1.5,-0.5])
du_max = np.array([1.5,0.5])

def simulate_path(SimLength):
    # initialize
    X = np.zeros((2,SimLength+1))
    # several step 
    '''interval = SimLength/7
    for i in range(SimLength):
        if i<interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        elif i<2*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,1])
        elif i<3*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        elif i<5*interval:# go north
            X[:,i+1] = X[:,i]+0.1*np.array([1,-1])
        elif i<6*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        else:
            X[:,i+1] = X[:,i]+0.1*np.array([1,1])'''
    interval = (SimLength-1)/6
    for i in range(SimLength):
        if i<interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        elif i<2*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,1])
        elif i<3*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([-1,1])
        elif i<4*interval:# go north
            X[:,i+1] = X[:,i]+0.1*np.array([-1,0])
        elif i<5*interval:# go north
            X[:,i+1] = X[:,i]+0.1*np.array([-1,-1])
        elif i<6*interval:
            X[:,i+1] = X[:,i]+0.1*np.array([1,-1])
        else:
            X[:,i+1]=X[:,i]+0.1*np.array([1,0])

        '''theta = -(2*(i+1)*pi/SimLength-pi/2)
        X[:,i+1] = np.array([np.cos(theta),np.sin(theta)-1])'''
        #X[:,i+1] = X[:,i]+0.1*np.array([1,0])
        X[:,i+1] = np.maximum(X[:,i+1],x_min)
        X[:,i+1] = np.minimum(X[:,i+1],x_max)
    return X

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
