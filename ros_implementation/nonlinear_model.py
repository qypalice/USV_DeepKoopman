from numpy import cos,sin
import numpy as np

#=========================================================================================
# discrete nonlinear model
#======================================================================================
def discrete_nonlinear(x,u,Ts):
    """
    discrete_nonlinear(x,u,Ts) returns the nonlinear discrete model
    x[k+1] = Ts*dx+x[k] of the state vector: 

    input vector    x  = [X, Y, psi]' position
                    u  = [v, w]' velocity
                    Ts    Sampling period
    output vector   y = x

    v     = surge velocity                    (m/s)     
    w     = yaw velocity                      (rad/s)
    X     = position in x-direction           (m)
    Y     = position in y-direction           (m)
    psi   = yaw angle                         (rad)

    matrix A = dx = [v*cos(psi)   v*sin(psi)    w]
    """
    # Inputs
    state = x.squeeze()
    vel = u.squeeze()
    v = vel[0]
    w = vel[1]
    psi = state[2]

    # Model matricses        
    dx = np.array([[v*cos(psi),v*sin(psi),w]])
    
    return Ts*dx+x