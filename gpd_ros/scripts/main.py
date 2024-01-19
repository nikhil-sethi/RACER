# from gym_pybullet_drones.envs import V
"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2,  PointField
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import subprocess
from scipy.spatial.transform import Rotation as R
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
import rospkg
import tf
import struct
import keyboard
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import PositionCommand
from scipy.spatial.transform import Rotation 

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = False
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 520
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def pack_rgb_image(rgb_image):
    height, width, _ = rgb_image.shape
    rgb_flat = rgb_image.reshape((height * width, 3)).astype(np.uint32)
    
    # creates unique integers from three rgb values using bitwise shifting operations
    packed_colors = (rgb_flat[:, 0] << 16) | (rgb_flat[:, 1] << 8) | rgb_flat[:, 2]
    return packed_colors.view(np.float32)


def create_point_cloud(depth_image, rgb_image, intrinsic_matrix):
    # Get height and width from depth image
    height, width = depth_image.shape

    # Create arrays of indices
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    indices = np.vstack((u.flatten(), v.flatten(), np.ones_like(u.flatten())))

    # Apply intrinsic matrix to get 3D coordinates
    coordinates_3d =  np.linalg.inv(intrinsic_matrix) @ indices 

    # Scale 3D coordinates by depth values
    coordinates_3d *= depth_image.flatten()
    
    # colour transformations

    rgb_packed = pack_rgb_image(rgb_image)
    
    point_cloud = np.column_stack((coordinates_3d.T, rgb_packed))
    
    # make far points invalid
    # point_cloud[coordinates_3d[-1, :]>999] = np.nan


    # Create a PointCloud2 message
    msg = PointCloud2()
    msg.header = rospy.Header(frame_id="drone_0/cam")

    msg.height = height
    msg.width = width
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    
    msg.data = np.asarray(point_cloud, np.float32).tobytes()
    msg.point_step = 16  # Size of one point in bytes (4 for x, 4 for y, 4 for z, 12 for rgb)
    msg.row_step = msg.point_step * width
    msg.is_dense = False  # Assuming there are no NaN values in the point cloud

    return msg

def img_publisher(pub, img, frame, encoding='passthrough'):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY ) 
    ros_image = bridge.cv2_to_imgmsg(img, encoding=encoding)
    ros_image.header.frame_id = frame
    pub.publish(ros_image)

# seg_timer = rospy.Timer(rospy.Duration(0.05), img_publisher)

def xacro2urdf(filename):
    rospack = rospkg.RosPack()
    path = rospack.get_path('gpd_ros')
    urdf_filename = os.path.join(filename.split('.')[0]+".urdf")
    urdf = open(urdf_filename, "w")
    subprocess.call(['rosrun', 'xacro', 'xacro', filename], stdout=urdf)
    return urdf_filename

class Env(CtrlAviary):
    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 1, neighbourhood_radius: float = np.inf, initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, pyb_freq: int = 240, ctrl_freq: int = 240, gui=False, record=False, obstacles=False, user_debug_gui=True, output_folder='results'):
        super().__init__(drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, pyb_freq, ctrl_freq, gui, record, obstacles, user_debug_gui, output_folder)
        self.br = tf.TransformBroadcaster()
        self.start = rospy.Time.now()
        self.cmd_subs = [rospy.Subscriber(f"planning/pos_cmd_{i+1}", PositionCommand, self.cmds_callback, callback_args=(i,)) for i in range(DEFAULT_NUM_DRONES)]
        self.target_xyzs = initial_xyzs.astype(np.float32)
        self.target_rpys = initial_rpys.astype(np.float32)
        self.target_rpy_rates = np.zeros((DEFAULT_NUM_DRONES, 3))
        self.target_vels = np.zeros((DEFAULT_NUM_DRONES, 3))

    def _addObstacles(self):
        """Remember to add separate urdfs for all the things you want to fake segmentation for"""
        # p.loadURDF(xacro2urdf("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_wall.xacro"),physicsClientId=self.CLIENT)
        # p.loadURDF(xacro2urdf("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_floor.xacro"),physicsClientId=self.CLIENT)
        # p.loadURDF(xacro2urdf("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_ceiling.xacro"),physicsClientId=self.CLIENT)
        # p.loadURDF("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_floor2.urdf",physicsClientId=self.CLIENT)
        # p.loadURDF("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_wall2.urdf",physicsClientId=self.CLIENT)
        p.loadSDF("/root/racer_ws/src/RACER/gpd_ros/urdf/gazebo/world2.sdf",physicsClientId=self.CLIENT)

    def publish_images(self, j):
        
        rgb, dep, seg = self._getDroneImages(j, segmentation=False)
        far = 1000
        print(np.max(dep))
        near = self.L+0.1
        dep = far * near / (far - (far - near) * dep)
        dep[dep>5] =0

        seg+=1 # because background is -1 and we turn it into 0
        """
        0: background
        1: pybullet ground plane
        2: drone 
        3: wall
        4: floor
        5: ceiling
        """
        
        # Create a colormap with enough colors for all unique indices. 
        cmap = plt.get_cmap('tab10', 5+DEFAULT_NUM_DRONES) # the number here will have to be set manualrly to fake unique colors for the segmentation data
        
        # Create a new image with unique colors for each segment
        seg = cmap(seg, bytes=True)

        img_publisher(seg_pubs[j], seg[...,:-1], f"drone_{j}/cam", "rgb8")
        img_publisher(rgb_pubs[j], rgb[...,:-1], f"drone_{j}/cam", "rgb8")
        img_publisher(dep_pubs[j], dep, f"drone_{j}/cam")

        cam_info_msg = CameraInfo()
        cam_info_msg.header = Header()
        # cam_info_msg.header.stamp = (rospy.Time.now()-self.start)
        cam_info_msg.header.frame_id = f"drone_{j}/odom"
        cam_info_msg.height = int(self.IMG_RES[1])
        # print(self.IMG_RES[1])
        cam_info_msg.width = int(self.IMG_RES[0])
        cam_info_msg.K = self.K.flatten()
        cam_info_msg.D = np.zeros(4)
        cam_info_msg.R = np.zeros(9)
        cam_info_msg.distortion_model = "radial-tangential"
        cam_info_msg.P = np.zeros((3,4))
        cam_info_msg.P[:3,:3] = self.K
        cam_info_msg.P = cam_info_msg.P.flatten()
        cam_info_pub.publish(cam_info_msg)

        # print(cv2.cvtColor(seg, cv2.COLOR_BGRA2RGB))
        # pcl = create_point_cloud(dep, seg[...,:-1], self.K)
        time.sleep(0.01)
        # pcl_pub.publish(pcl)

    def publish_states(self, j):
        # Pose 
        pos = self.pos[j, :]
        # print(pos)
        quat = self.quat[j, :]
        
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        # pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"
        pose_msg.pose = Pose(position=Point(pos[0],pos[1],pos[2]), orientation = Quaternion(quat[0], quat[1], quat[2], quat[3]))
        pose_pubs[j].publish(pose_msg)

        # transform for the drone (need this for the point cloud)
        self.br.sendTransform(pos, 
                              quat,
                              rospy.Time.now(),
                              f"drone_{j}/odom",
                              "floor")
        # odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()-rospy.Duration(0.2)
        odom_msg.header.frame_id = "world"
        vel = self.vel[j,:]
        odom_msg.pose.pose.position.x = pos[0]
        odom_msg.pose.pose.position.y = pos[1]
        odom_msg.pose.pose.position.z = pos[2]
        odom_msg.twist.twist.linear.x = vel[0]
        odom_msg.twist.twist.linear.y = vel[1]
        odom_msg.twist.twist.linear.z = vel[2]
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        odom_pubs[j].publish(odom_msg)                     


        # camera transform
        pos = self.T[:,3]
        self.T[:3,1:3] *= -1 # this hack is needed to convert from pybullet to ros/tf format.
        quat = tf.transformations.quaternion_from_matrix(self.T)
        self.br.sendTransform(pos, 
                              quat,
                              rospy.Time.now(),
                              f"drone_{j}/cam",
                              f"floor")   
        campose_msg = PoseStamped()
        campose_msg.header = Header()
        # pose_msg.header.stamp = rospy.Time.now()
        campose_msg.header.frame_id = "world"
        campose_msg.pose = Pose(position=Point(pos[0],pos[1],pos[2]), orientation = Quaternion(quat[0], quat[1], quat[2], quat[3]))
        campose_pubs[j].publish(campose_msg)

    def cmds_callback(self, msg, args):
        j = args[0]
        # p_t = np.array([msg.position.x,msg.position.y,msg.position.z])
        # v = p_t - self.pos[j, :]
        # self.cmds[j,:3] = v/np.abs(v)
        # self.cmds[j,3] = min(1,np.linalg.norm(v))
        # print(self.cmds[j])
        self.target_xyzs[j] = np.array([msg.position.x,msg.position.y,msg.position.z])
        # print(msg.yaw)
        self.target_rpys[j] = np.array([0, 0, msg.yaw])
        self.target_rpy_rates[j] = np.array([0, 0, msg.yaw_dot])
        self.target_vels[j] = np.array([msg.velocity.x,msg.velocity.y,msg.velocity.z])
    
def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    # INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    # print(INIT_XYZS)
    INIT_XYZS = np.array([[0,-0.1,0.3]])
    # print(INIT_XYZS)
    INIT_RPYS = np.array([[0, 0, 0] for i in range(num_drones)])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/ +0.0005 ——— 
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
    env = Env(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Make it rain rubber ducks #############################
        # if i/env.SIM_FREQ>5 and i%10==0 and i/env.SIM_FREQ<10: p.loadURDF("duck_vhacd.urdf", [0+random.gauss(0, 0.3),-0.5+random.gauss(0, 0.3),3], p.getQuaternionFromEuler([random.randint(0,360),random.randint(0,360),random.randint(0,360)]), physicsClientId=PYB_CLIENT)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        
        #### Compute control for the current way point #############
        for j in range(num_drones):
        
            # if keyboard.is_pressed('down'):
            #     action[j,:] = [-1,0,0,1]
            # elif keyboard.is_pressed('up'):
            #     action[j,:] = [1,0,0,1]
            # elif keyboard.is_pressed('left'):
            #     action[j,:] = [0,1,0,1]
            # elif keyboard.is_pressed('right'):
            #     action[j,:] = [0,-1,0,1]
            # else:

            action[j,:],_,_ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                    state=obs[j],
                                                    target_pos=env.target_xyzs[j],
                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                    target_vel = env.target_vels[j],
                                                    target_rpy_rates = env.target_rpy_rates[j],
                                                    target_rpy=env.target_rpys[j]
                                                    )
            # action[j,:] = env.cmds[j]
            # env.pos[j] = env.target_xyzs[j]
            # print("dfgadfg")
            # print(env.target_rpys[j])
            # print(action[j])
            # env.quat[j] = Rotation.from_euler('z',env.target_rpys[j,2]).as_quat()
            start = time.time()
            env.publish_images(j)
            # print(time.time()-start)       
            env.publish_states(j)
                 

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([TARGET_POS[wp_counters[j], 0:2], INIT_XYZS[j, 2], INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()




if __name__ == "__main__":
    rospy.init_node("gpd_main", disable_signals=True)
    seg_pubs = [rospy.Publisher(f"drone_{i}/img_seg", Image, queue_size=1) for i in range(DEFAULT_NUM_DRONES)]
    rgb_pubs = [rospy.Publisher(f"drone_{i}/img_rgb", Image, queue_size=1) for i in range(DEFAULT_NUM_DRONES)]
    dep_pubs = [rospy.Publisher(f"drone_{i}/img_dep", Image, queue_size=1) for i in range(DEFAULT_NUM_DRONES)]

    pose_pubs = [rospy.Publisher(f"drone_{i}/pose", PoseStamped, queue_size=1) for i in range(DEFAULT_NUM_DRONES)]
    campose_pubs = [rospy.Publisher(f"drone_{i}/camera", PoseStamped, queue_size=1) for i in range(DEFAULT_NUM_DRONES)]
    odom_pubs = [rospy.Publisher(f"drone_{i}/odom", Odometry, queue_size=10) for i in range(DEFAULT_NUM_DRONES)]
    cam_info_pub = rospy.Publisher(f"drone_0/camera_info", CameraInfo, queue_size=1)
    pcl_pub = rospy.Publisher(f"drone_0/semantic_pointcloud", PointCloud2, queue_size=10)


    bridge = CvBridge()

    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS, _ = parser.parse_known_args()

    run(**vars(ARGS))

