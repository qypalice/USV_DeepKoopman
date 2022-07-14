import sys
import cvxpy as cp
import numpy as np
from numpy import pi
from numpy.linalg import matrix_power
import matplotlib.pyplot as plt
from nonlinear_model import discrete_nonlinear
from Koopman_numpy import Koopman_numpy
import time
from tqdm import tqdm

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
    # print(path)
    print(sim)

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

'''Build robust MPC for tracking using Koopman operators'''
# by Ye Wang, since 1st June 2022

def KTMPC_optimiser(z0,zt,zrange,urange,A,B,Q,R,S,N):

    # Get dimensions
    nz = A.shape[0]
    nu = B.shape[1]

    # T = 200*np.eye(nz)

    # Declare decision variables
    z = cp.Variable((nz,N+1))
    u = cp.Variable((nu,N))
    zs = cp.Variable((nz))
    us = cp.Variable((nu))

    # Define object function
    obj = cp.quad_form(zs.T-zt,S) #+ cp.quad_form(zs.T-z[:,-1],T)

    usmin = np.array([-0.1, -urange[1]])
    umin = np.array([0.1, -urange[1]])

    # Define constraints
    cons = [z[:,0]==z0,z[:,-1]==zs]
    cons += [zs == A @ zs + B @ us,
             us >= usmin, us <= urange]
             # zs >= -zrange, zs <= zrange,
             # us >= -urange, us <= urange]

    for t in range(N):
        obj += cp.quad_form(z[:,t]-zs,Q) + cp.quad_form(u[:,t]-us,R)
        cons += [z[:,t+1] == A@z[:,t] + B@u[:,t],
                 u[:, t] >= umin, u[:, t] <= urange]
                 # z[:,t]>=-zrange, z[:,t]<=zrange,
                 # u[:,t]>=-urange, u[:,t]<=urange]

    # Solve optimisation problem
    prob = cp.Problem(cp.Minimize(obj),cons)
    prob.solve(solver=cp.OSQP,eps_abs=1e-2,verbose=False)

    # print(prob.status)
    uopt = u[:,0].value
    zsopt = zs.value
    usopt = us.value
    flag = prob.status

    return uopt, zsopt, usopt, flag

def KTMPC_simulation(model_file,yt,init_state,xrange,urange,q,r,s,N,simtime,changedt,model_flag):

    # load model
    operator = Koopman_numpy(model_file)
    A, B = operator.linear_matrix()
    nz = A.shape[0]
    nu = B.shape[1]

    # Prepare weighting matrices
    # Q = q * np.eye(nz)
    Q = np.diag(q)
    # R = r * np.eye(nu)
    R = np.diag(r)
    # S = s * np.eye(nz)
    S = np.diag(s)

    # Find lifted references
    zt = operator.encode(yt[0])
    yst = yt[0]

    # print(zt)
    # print(yst)
    zrange = operator.encode(xrange)

    # Initialisation
    xk = init_state
    zk = operator.encode(xk)

    # Save data
    x = xk
    z = zk
    zr = zt
    yr = yst

    j=0

    print('Simulation in progress:')
    pbar = tqdm(total=simtime)
    for i in range(simtime):
        # print(i)
        # Solve KTMPC optimisation problem
        uk, zsk, usk,flag = KTMPC_optimiser(zk, zt, zrange, urange, A, B, Q, R, S, N)
        # print(zsk)
        # print(zsk)
        # print(flag)

        if flag not in ["infeasible","unbounded"]:
            ysk = operator.decode(zsk)
            if i == 0:
                u = uk
                zs = zsk
                ys = ysk
                us = usk
            else:
                u = np.c_[u,uk]
                zs = np.c_[zs,zsk]
                ys = np.c_[ys,ysk]
                us = np.c_[us,usk]

            if model_flag == "nonlinear":
                # Apply control to nonlinear system
                xk = discrete_nonlinear(xk, uk, Ts).squeeze()
                zk = operator.encode(xk)
            else:
                # Apply control to Koopman linear model
                zk = operator.linear(zk,uk)
                xk = operator.decode(zk)

            # Change reference point
            # print(np.linalg.norm(xk[0:1] - ysk[0:1]))
            # if np.linalg.norm(xk[0]-yt[j,0]) < 0.1 and np.linalg.norm(xk[1]-yt[j,1]) < 0.1:
            # if np.linalg.norm(xk[0:2] - yt[j,0:2]) <= 0.1:
            if i>0 and i%changedt == 0:
                j += 1
                zt = operator.encode(yt[j])
                yst = yt[j]

            # Save Data
            x = np.c_[x,xk]
            z = np.c_[z,zk]
            zr = np.c_[zr,zt]
            yr = np.c_[yr,yst]

        else:
            print("Infeasible problem")
            break
        pbar.update(1)
    pbar.close()
    print('Simulation Done!')
    return x,z,u,zr,yr,zs,ys,us

def ComputeTheta(yk,yt):
    y = yt[1]-yk[1]
    x = yt[0]-yk[0]

    theta = np.arctan2(y,x)
    if y>0 and x<0:
        theta = theta - 2*np.pi
    return theta

def KTMPC_simulation2(model_file,yt,init_state,xrange,urange,q,r,s,N,simtime,changedt,model_flag):

    # load model
    operator = Koopman_numpy(model_file)
    A, B = operator.linear_matrix()
    nz = A.shape[0]
    nu = B.shape[1]

    # Prepare weighting matrices
    # Q = q * np.eye(nz)
    Q = np.diag(q)
    # R = r * np.eye(nu)
    R = np.diag(r)
    # S = s * np.eye(nz)
    S = np.diag(s)

    # Find lifted references
    zt = operator.encode(yt[0])
    yst = yt[0]
    ypr = yt[0,0:2]

    # print(zt)
    # print(yst)
    zrange = operator.encode(xrange)

    # Initialisation
    xk = init_state
    zk = operator.encode(xk)

    # Save data
    x = xk
    z = zk
    zr = zt
    yr = yst

    j=0
    xpk = xk

    print('Simulation in progress:')
    pbar = tqdm(total=simtime)
    for i in range(simtime):
        # print(i)
        # Solve KTMPC optimisation problem
        uk, zsk, usk,flag = KTMPC_optimiser(zk, zt, zrange, urange, A, B, Q, R, S, N)

        if flag not in ["infeasible","unbounded"]:
            ysk = operator.decode(zsk)
            if i == 0:
                u = uk
                zs = zsk
                ys = ysk
                us = usk
            else:
                u = np.c_[u,uk]
                zs = np.c_[zs,zsk]
                ys = np.c_[ys,ysk]
                us = np.c_[us,usk]

                xpk = xk

            if model_flag == "nonlinear":
                # Apply control to nonlinear system
                xk = discrete_nonlinear(xk, uk, Ts).squeeze()
                zk = operator.encode(xk)
            else:
                # Apply control to Koopman linear model
                zk = operator.linear(zk,uk)
                xk = operator.decode(zk)

            # Change reference point
            # print(np.linalg.norm(xk[0:1] - ysk[0:1]))
            # if np.linalg.norm(xk[0]-yt[j,0]) < 0.1 and np.linalg.norm(xk[1]-yt[j,1]) < 0.1:
            if np.linalg.norm(xk[0:2] - yt[j,0:2]) <= 0.1: #or np.linalg.norm(xk[0:2] - xpk[0:2]) <= 0.001:
            # if i>0 and i%changedt == 0:
                j += 1
                ypr = yt[j,0:2]

            # Update theta reference
            thetar = ComputeTheta(xk, ypr)
            ytk = np.hstack((ypr, thetar))
            zt = operator.encode(ytk)
            yst = ytk

            # Save Data
            x = np.c_[x,xk]
            z = np.c_[z,zk]
            zr = np.c_[zr,zt]
            yr = np.c_[yr,yst]

        else:
            print("Infeasible problem")
            break
        pbar.update(1)
    pbar.close()
    print('Simulation Done!')
    return x,z,u,zr,yr,zs,ys,us

def KTMPC_simulation_plot(x,z,u,yt,zr,yr,ys,us,urange,simtime,Ts):

    tseries = Ts*np.linspace(1,simtime,simtime)
    tseries1 = Ts*np.linspace(0, simtime, simtime+1)
    # xr1 = yr[0,:] #yt[0]*np.ones(simtime+1)
    # xr2 = yr[1,:] #yt[1]*np.ones(simtime+1)
    # xr3 = yr[2,:] #yt[2]*np.ones(simtime+1)

    uub1 = urange[0]*np.ones((1,simtime))
    uub2 = urange[1] * np.ones((1, simtime))
    uub = np.vstack((uub1,uub2))
    ulb = -uub

    # plt.figure()
    fig,axs = plt.subplots(3,1)
    axs[0].plot(tseries1, yr[0,:], color='green', linestyle = 'dashed',label='reference')
    axs[0].plot(tseries, ys[0,:], 'r-.', label='steady output')
    axs[0].plot(tseries1, x[0,:], color='black',label='KTMPC')
    axs[0].set_ylabel(r'$x$')
    axs[0].legend()

    axs[1].plot(tseries1, yr[1,:], color='green', linestyle='dashed', label='reference')
    axs[1].plot(tseries, ys[1, :], 'r-.', label='steady output')
    axs[1].plot(tseries1, x[1,:], color='black', label='KTMPC')
    axs[1].set_ylabel(r'$y$')

    axs[2].plot(tseries1, yr[2,:], color='green', linestyle='dashed', label='reference')
    axs[2].plot(tseries, ys[2, :], 'r-.', label='steady output')
    axs[2].plot(tseries1, x[2,:], color='black', label='KTMPC')
    axs[2].set_xlabel('time [s]')
    axs[2].set_ylabel(r'$\theta$')

    fig.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2)
    axs[0].plot(tseries, uub[0,:], 'y-.',label='input constraint')
    axs[0].plot(tseries, ulb[0, :], 'y-.')
    axs[0].plot(tseries, us[0,:], color='green', linestyle='dashed', label='steady input')
    axs[0].plot(tseries, u[0, :], color='black', label='KTMPC')
    axs[0].set_ylabel(r'$v$')
    axs[0].legend()

    axs[1].plot(tseries, uub[1, :], 'y-.', label='input constraint')
    axs[1].plot(tseries, ulb[1, :], 'y-.')
    axs[1].plot(tseries, us[1,:], color='green', linestyle='dashed', label='steady input')
    axs[1].plot(tseries, u[1, :], color='black', label='KTMPC')
    axs[1].set_ylabel(r'$w$')

    fig.tight_layout()
    plt.show()

    # Plot 2D figure
    plt.figure()
    plt.plot(yt[:,0], yt[:,1], 'bx')
    # plt.plot(yst[0],yst[1],color='tab:red', marker='o')
    # plt.plot(ys[0], ys[1], 'go')
    plt.plot(x[0, :], x[1, :], color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    # plt.tight_layout()
    plt.show()


def KTMPC_analysis_z_plot(x,z,u,yt,zr,yr,zs,ys,simtime,Ts):

    tseries = Ts*np.linspace(1,simtime,simtime)
    tseries1 = Ts*np.linspace(0, simtime, simtime+1)

    nz = int(z.shape[0])
    ns = int(nz/3)+1

    # Plot
    # fig,axs = plt.subplots(ns,2)
    plt.figure()
    for i in range(nz):
        plt.subplot(ns,3,i+1)
        plt.plot(tseries1,zr[i,:], color='green', linestyle = 'dashed',label='reference z')
        plt.plot(tseries,zs[i,:], color='red', linestyle='dashed', label='steady z')
        plt.plot(tseries1,z[i,:], color='black',label='KTMPC')
        plt.title(str(i))
        plt.tight_layout()
        # axs[i].set_ylabel(r'label')

    plt.legend()
    # fig.tight_layout()
    plt.show()


