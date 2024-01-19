from gpd_ros.env import Env
import rospy
import gpd_ros.conf as conf
import time
import numpy as np
from gpd_ros.sensors import Sensors, Timer
from nav_msgs.msg import Odometry
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from quadrotor_msgs.msg import PositionCommand
import tf

class Simulation:
    def __init__(self):
        rospy.init_node("simulation")
        self.env = Env(obstacles=conf.DEFAULT_OBSTACLES, 
                        initial_xyzs=conf.INIT_XYZS, 
                        initial_rpys=conf.INIT_RPYS, 
                        ctrl_freq=conf.DEFAULT_CONTROL_FREQ_HZ, 
                        gui=conf.DEFAULT_GUI, 
                        user_debug_gui=conf.DEFAULT_USER_DEBUG_GUI,
                        record=conf.DEFAULT_RECORD_VISION,
                        physics=conf.DEFAULT_PHYSICS,
                        num_drones=conf.DEFAULT_NUM_DRONES,
                        drone_model=conf.DEFAULT_DRONES
                        )

        self.controller = DSLPIDControl(drone_model=conf.DEFAULT_DRONES)
        self.sensors = Sensors(self.env, self.env.L)
        

        self.target_xyzs = conf.INIT_XYZS.astype(np.float32)
        self.target_rpys = conf.INIT_RPYS.astype(np.float32)
        self.target_rpy_rates = np.zeros((conf.DEFAULT_NUM_DRONES, 3))
        self.target_vels = np.zeros((conf.DEFAULT_NUM_DRONES, 3))
        self.actions = np.zeros((conf.DEFAULT_NUM_DRONES, 4))

        self.cmd_subs = [rospy.Subscriber(f"planning/pos_cmd_{i+1}", PositionCommand, self.cmds_callback, callback_args=(i,)) for i in range(conf.DEFAULT_NUM_DRONES)]
        self.odom_pubs = [rospy.Publisher(f"drone_{i}/odom", Odometry, queue_size=10) for i in range(conf.DEFAULT_NUM_DRONES)]

        self.start = time.time()
        self.counter = 0
        # self.env.reset()
        sim_dt = 1/conf.DEFAULT_SIMULATION_FREQ_HZ
        self.br = tf.TransformBroadcaster()
        # sim_dt = 1/100
        self.sim_timer = rospy.Timer(rospy.Duration(sim_dt), self.sim_update)
        rospy.spin()
        

    def cmds_callback(self, msg, args):
        j = args[0]
        self.target_xyzs[j] = np.array([msg.position.x,msg.position.y,msg.position.z])
        self.target_rpys[j] = np.array([0, 0, msg.yaw])
        self.target_rpy_rates[j] = np.array([0, 0, msg.yaw_dot])
        self.target_vels[j] = np.array([msg.velocity.x,msg.velocity.y,msg.velocity.z])
    
    def update_odom(self, j):
        # Pose 
        self.pos = self.env.pos[j, :]
        self.quat = self.env.quat[j, :]
        self.vel = self.env.vel[j,:]
        self.ang_v = self.env.ang_v[j,:]
        # odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()-rospy.Duration(0.2)
        odom_msg.header.frame_id = "world"
        
        odom_msg.pose.pose.position.x = self.pos[0]
        odom_msg.pose.pose.position.y = self.pos[1]
        odom_msg.pose.pose.position.z = self.pos[2]
        odom_msg.twist.twist.linear.x = self.vel[0]
        odom_msg.twist.twist.linear.y = self.vel[1]
        odom_msg.twist.twist.linear.z = self.vel[2]
        odom_msg.twist.twist.angular.x = self.ang_v[0]
        odom_msg.twist.twist.angular.y = self.ang_v[1]
        odom_msg.twist.twist.angular.z = self.ang_v[2]
        odom_msg.pose.pose.orientation.x = self.quat[0]
        odom_msg.pose.pose.orientation.y = self.quat[1]
        odom_msg.pose.pose.orientation.z = self.quat[2]
        odom_msg.pose.pose.orientation.w = self.quat[3]

        self.odom_pubs[j].publish(odom_msg)    

        self.br.sendTransform(self.pos, 
                              self.quat,
                              rospy.Time.now(),
                              f"drone_{j}/odom",
                              f"world")  

        return odom_msg
    

    def sim_update(self, event):
        if time.time()-self.start > conf.DEFAULT_DURATION_SEC:
            self.sim_timer.shutdown()
        
        obs, reward, terminated, truncated, info = self.env.step(self.actions)
        
        #### Compute control for the current way point #############
        for j in range(conf.DEFAULT_NUM_DRONES):
        
            # if keyboard.is_pressed('down'):
            #     action[j,:] = [-1,0,0,1]
            # elif keyboard.is_pressed('up'):
            #     action[j,:] = [1,0,0,1]
            # elif keyboard.is_pressed('left'):
            #     action[j,:] = [0,1,0,1]
            # elif keyboard.is_pressed('right'):
            #     action[j,:] = [0,-1,0,1]
            # else:

            self.actions[j,:],_,_ = self.controller.computeControlFromState(control_timestep=self.env.CTRL_TIMESTEP,
                                                    state=obs[j],
                                                    target_pos=self.target_xyzs[j],
                                                    # target_pos=INIT_XYZS[j, :] + TARGET_POS[wp_counters[j], :],
                                                    target_vel = self.target_vels[j],
                                                    # target_rpy_rates = self.target_rpy_rates[j],
                                                    target_rpy=self.target_rpys[j]
                                                    )
            start = time.time()
            self.update_odom(j)

        # self.sensors.update_images()

        #### Printout ##############################################
        # self.env.render()
        self.counter += 1

        #### Sync the simulation ###################################
        if conf.DEFAULT_GUI:
            sync(self.counter, self.start, self.env.CTRL_TIMESTEP)

if __name__=="__main__":
    
    sim = Simulation()
    