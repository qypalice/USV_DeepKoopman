import rospy
from geometry_msgs.msg import PoseStamped,Twist
from zed_interfaces.srv import set_pose
from numpy import pi
from MPC import *
from Koopman_numpy import Koopman_numpy
import time

def path_process(ref):
    diff = ref[:,1:]-ref[:,:-1]
    angle = np.arctan(diff[1,:]/diff[0,:])
    for i in range(angle.shape[0]-1):
        while angle[i+1]-angle[i]>pi/2:
            angle[i+1] -= pi
        while angle[i+1]-angle[i]<-pi/2:
            angle[i+1] += pi
    ref = np.r_[ref[:,1:],np.array([angle])]
    temp = np.zeros((3,0))
    temp = np.c_[temp,ref[:,0]]
    for i in range(1,ref.shape[1]):
        while abs(ref[2,i]-temp[2,-1])>0.03:
            step = np.array([temp[0,-1],temp[1,-1],temp[2,-1]+0.03*np.sign(ref[2,i]-temp[2,-1])])
            temp = np.c_[temp,step]
        if ref[2,i]-temp[2,-1] !=0:
            temp = np.c_[temp,[temp[0,-1],temp[1,-1],ref[2,i]]]
        while abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1])>0.05:
            ratio_x = (ref[0,i]-temp[0,-1])/(abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]))
            ratio_y = (ref[1,i]-temp[1,-1])/(abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]))
            step = np.array([temp[0,-1]+0.05*ratio_x,temp[1,-1]+0.05*ratio_y,temp[2,-1]])
            temp = np.c_[temp,step]
        if abs(ref[0,i]-temp[0,-1])+abs(ref[1,i]-temp[1,-1]) !=0:
            temp = np.c_[temp,ref[:,i]]
    ref = temp
    for i in range(ref.shape[1]):
        while ref[2,i]>pi+0.05:
            ref[2,i] -= 2*pi
        while ref[2,i]<-pi/2:
            ref[2,i] += pi
        while ref[2,i]<-0.05-pi:
            ref[2,i] += 2*pi
        while ref[2,i]>pi/2:
            ref[2,i] -= pi
    # lift the reference
    lifted_ref = np.zeros(11*ref.shape[1])
    for i in range(ref.shape[1]):
        lifted_ref[11*i:11*i+11] = operater.encode(ref[:,i])
    return ref,lifted_ref

#=======================================================================
#global all variables & parameters
# set parameters,path,limitation for solver
global Qbig,H,Gamma,Theta,rho,operater,u_range,Nc
Ts = 0.1
Q = 100*np.diag(np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]))
R = 0.1*np.diag(np.array([2.,1.]))
rho = 0.01
Nc = 5
thre = 0.3
u_init = np.array([0.,0.])
x_init = np.array([0.,0.,0.])
model_file = 'unmasked_encoder_[3, 32, 64, 8]_decoder_[11, 128, 64, 32, 3]_hyper_[1.0, 3.0, 0.3, 1e-07, 1e-07, 1e-07, 10]_batch_10'
operater = Koopman_numpy(model_file)
A,B = operater.linear_matrix()
Gamma,Theta,Qbig,H,u_range = get_Augmented_Matrix(A,B,Q,R,rho,Nc)
#================================
# get path
ref = simulate_path(46)
global lifted_ref,path_ref,i,j
ref,lifted_ref = path_process(ref)
i = 0
j = 0
path_ref = lifted_ref[11*i:11*i+11*Nc]
#================================
#record path
global ref_arg, path, x, u
ref_arg = np.zeros((3,1))
path = np.zeros((3,1))
x = x_init
u = u_init
#========================================================================
# Subscriber + timer for 10 Hz update rate
def reset_client(x,y,z,roll,pitch,yaw):
    rospy.wait_for_service('/zed2i/zed_node/set_pose')
    pose_set = rospy.ServiceProxy('/zed2i/zed_node/set_pose', set_pose)
    pose_set(x,y,z,roll,pitch,yaw)
def control_node():
    reset_client(0,0,0,0,0,0)
    rospy.init_node('control', anonymous=True)
    rospy.Subscriber("/mavros/vision_pose/pose", PoseStamped, callback)
    timer = rospy.Timer(rospy.Duration(0.1), timer_callback)
    rospy.spin()

def callback(msg):
    global pose
    pose = msg

# Implement the controller in the timer callback block
def timer_callback(event):
    global i,j,u,path,ref_arg,path_ref
    #====transform input position
    x[0] = pose.pose.position.x
    x[1] = pose.pose.position.y
    x[2] = np.arctan2(2 * (pose.pose.orientation.w * pose.pose.orientation.z + pose.pose.orientation.x * pose.pose.orientation.y), 
                    1 - 2 * (pose.pose.orientation.y * pose.pose.orientation.y + pose.pose.orientation.z * pose.pose.orientation.z))
    #====get optimized velocity
    vel = Twist()
    if x[0]**2+x[1]**2 <20:
        if np.linalg.norm(x-ref[:,i])<thre or j>10:
            j = 0
            i += 1
            path_ref = lifted_ref[11*i:11*i+11*Nc]
        else:
            j += 1
        u = MPC_solver_aug(Qbig,H,Gamma,Theta,rho,path_ref,operater.encode(x),u,u_range,Nc)
        vel.linear.y = u[0]
        vel.angular.z = u[1]
    # publish the node
    pub.publish(vel)
    #path = np.c_[path,x]
    #ref_arg = np.c_[ref_arg,ref[:,i]]
    print("Step "+str(i)+":"+str(x)+", input:"+str(u))
    if i >30-Nc:
        print("Path finished!")

# Pulisher
# Change this to publish velocity by changing the topic name and the data type
# It publish position at 10Hz at the moment
pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=10)
#pub = rospy.Publisher('/input_unstamped', Twist, queue_size=10)

# Process the path - generate angle, seperate path into smaller steps

# Main loop
if __name__ == '__main__':

    # start control node
    try:
        control_node()

    except KeyboardInterrupt:
        print("Shuting down the controller")

    # plot
    #MPC_process_plot(ref_arg,path,path.shape[1],lifted=False)
