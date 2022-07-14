import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_model import discrete_nonlinear
from Koopman_numpy import Koopman_numpy
from tqdm import tqdm


# define global parameters
Ts = 0.1
x_min = np.array([-3,-3])
x_max = np.array([3,3])
u_min = np.array([-0.3,-0.3])
u_max = np.array([0.3,0.3])
du_min = np.array([-1.5,-0.5])
du_max = np.array([1.5,0.5])
x_range = np.array([2.,2.,np.pi])
u_range = np.array([0.4,0.5])

def ComputeTheta(yk,yt):
    y = yt[1]-yk[1]
    x = yt[0]-yk[0]

    theta = np.arctan2(y,x)
    if y>0 and x<0:
        theta = theta - 2*np.pi
    return theta
    
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


