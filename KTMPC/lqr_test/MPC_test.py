import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped,Twist
from zed_interfaces.srv import set_pose
from MPC import KTMPC_optimiser
from Koopman_numpy import Koopman_numpy
import time

def ComputeTheta(yk,yt):
    y = yt[1]-yk[1]
    x = yt[0]-yk[0]

    theta = np.arctan2(y,x)
    if y>0 and x<0:
        theta = theta - 2*np.pi
    return theta
#=======================================================================
#global all variables & parameters
# set parameters,path,limitation for solver
global Q,R,S,N,operater,zrange,urange
Ts = 0.1
model_file = 'unmasked_encoder_[3, 32, 64, 8]_decoder_[11, 128, 64, 32, 3]_hyper_[1.0, 3.0, 0.3, 1e-07, 1e-07, 1e-07, 10]_batch_10'
operater = Koopman_numpy(model_file)
A,B = operater.linear_matrix()
# Weights
q = 10*np.array([10,10,10,10,10,10,10,10,10,10,10])
r = 0.01*np.array([1,1]) # for u
s = 100*np.array([10,10,10,10,10,10,10,10,10,10,10]) # for offset function
Q = np.diag(q)
R = np.diag(r)
S = np.diag(s)
# Prediction horizon
N = 10
xrange = np.array([2.,2.,np.pi])
urange = np.array([0.4,0.5])
zrange = operater.encode(xrange)
init_state = np.array([0.,0.,np.pi/4])
#================================
# get path
global i,j,x,yt,zt,simtime
x = init_state
yt = np.array([[.5,.5,0.],[1.,1.,0],[2.,1.,0.],[2.5,0.5,0.],[3.,0.,0.],[2.5,-0.5,0.],[2.,-1.,0.],[1.,-1.,0.],[0.,0.,0.]])
simtime =  540 #changedt*yt.shape[0]
# Find lifted references
zt = operater.encode(yt[0])
yst = yt[0]
ypr = yt[0,0:2]
i = 0
j = 0
#================================
#record path, add if needed

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
    #====transform input position
    x[0] = pose.pose.position.x
    x[1] = pose.pose.position.y
    x[2] = np.arctan2(2 * (pose.pose.orientation.w * pose.pose.orientation.z + pose.pose.orientation.x * pose.pose.orientation.y), 
                    1 - 2 * (pose.pose.orientation.y * pose.pose.orientation.y + pose.pose.orientation.z * pose.pose.orientation.z))
    #====get optimized velocity
    vel = Twist()
    zk = operater.encode(x)
    u, zsk, usk,flag = KTMPC_optimiser(zk, zt, zrange, urange, A, B, Q, R, S, N)
    vel.linear.y = u[0]
    vel.angular.z = u[1]
    # publish the node
    pub.publish(vel)
    i += 1
    if np.linalg.norm(x[0:2] - yt[j,0:2]) <= 0.1: #or np.linalg.norm(xk[0:2] - xpk[0:2]) <= 0.001:
        j += 1
        ypr = yt[j,0:2]
    # Update theta reference
    thetar = ComputeTheta(x, ypr)
    ytk = np.hstack((ypr, thetar))
    zt = operater.encode(ytk)
    print("Step "+str(i)+":"+str(x)+", input:"+str(u))
    if i >simtime-N:
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
