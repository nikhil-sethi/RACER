import pybullet as p
import numpy as np
import rospy
import gpd_ros.conf as conf
from sensor_msgs.msg import Image, CameraInfo, PointCloud2,  PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, Pose
from cv_bridge import CvBridge
import cv2
import tf
import time
import pybullet_data
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
import threading
import matplotlib.pyplot as plt

class Timer(rospy.Timer):
    """Custom class to get around EGL problems"""
    def __init__(self, env, period, callback, oneshot=False, reset=False):
        self.env = env
        super().__init__(period,callback, oneshot=oneshot, reset=reset)

    def run(self):
        # the following egl code and environment reset HAS to be done in this thread otherwise things dont work with EGL (which is very fast with p.DIRECT)
        if not conf.DEFAULT_GUI:
            if (egl):
                pluginId = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
            else:
                pluginId = p.loadPlugin("eglRendererPlugin")
            print("pluginId=",pluginId)
        self.env.reset()
        super().run()


def create_point_cloud(depth_image, seg_mask, intrinsic_matrix):
    indices = np.vstack([seg_mask[:,1],seg_mask[:,0], np.ones(len(seg_mask))])

    # Apply intrinsic matrix to get 3D coordinates
    coordinates_3d =  np.linalg.inv(intrinsic_matrix) @ indices

    # Scale 3D coordinates by depth values
    point_cloud = (coordinates_3d * depth_image.flatten()).T
    # print(point_cloud)
    # colour transformations

    # rgb_packed = pack_rgb_image(rgb_image)
    
    # point_cloud = np.column_stack((coordinates_3d.T, rgb_packed))
    
    # make far points invalid
    # point_cloud[coordinates_3d[-1, :]>999] = np.nan


    # Create a PointCloud2 message
    msg = PointCloud2()
    msg.header = rospy.Header(frame_id="drone_0/camera")

    msg.height = 1
    msg.width = len(point_cloud)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    
    msg.data = np.asarray(point_cloud, np.float32).tobytes()
    msg.point_step = 12  # Size of one point in bytes (4 for x, 4 for y, 4 for z)
    msg.row_step = msg.point_step * len(point_cloud)
    msg.is_dense = False  # Assuming there are no NaN values in the point cloud

    return msg

class Sensors:
    def __init__(self, env, L) -> None:
        # rospy.init_node("sensors")

        self.num_drones = conf.DEFAULT_NUM_DRONES
        self.L = L
        self.env = env
        self.CLIENT = env.CLIENT
        
        self.seg_pubs = [rospy.Publisher(f"drone_{i}/img_seg", Image, queue_size=1) for i in range(self.num_drones)]
        self.cam_info_pub = rospy.Publisher(f"drone/camera_info", CameraInfo, queue_size=1)

        # rgb_pubs = [rospy.Publisher(f"drone_{i}/img_rgb", Image, queue_size=1) for i in range(DEFAULT_NUM_DRONES)]
        self.dep_pubs = [rospy.Publisher(f"drone_{i}/img_dep", Image, queue_size=1) for i in range(self.num_drones)]
        self.sensor_pose_pubs = [rospy.Publisher(f"drone_{i}/camera", PoseStamped, queue_size=1) for i in range(self.num_drones)]
        self.pcl_pub = rospy.Publisher(f"drone_0/sem_cloud", PointCloud2, queue_size=10)

        self.cam_views = np.zeros((self.num_drones,16))
        self.cam_projs = np.zeros((self.num_drones,16))
        self.br = tf.TransformBroadcaster()
        # self.odom_subs = [rospy.Subscriber(f"drone_{i}/odom", Odometry, self.update_views, callback_args=(i,)) for i in range(conf.DEFAULT_NUM_DRONES)]
        self.bridge = CvBridge()

        self.img_timers = [Timer(env, rospy.Duration(0.1), self.update_images) for i in range(self.num_drones)]
        
        # rospy.spin()

    def publish_img(self, pub, img, encoding='passthrough'):
        ros_image = self.bridge.cv2_to_imgmsg(img, encoding=encoding)
        ros_image.header.frame_id = "drone_0/camera"
        pub.publish(ros_image)

    def publish_caminfo(self, j):
        cam_info_msg = CameraInfo()
        # cam_info_msg.header = Header()
        # cam_info_msg.header.stamp = (rospy.Time.now()-self.start)
        cam_info_msg.header.frame_id = f"drone_{j}/camera"
        cam_info_msg.height = int(conf.IMG_RES[1])
        # print(self.IMG_RES[1])
        cam_info_msg.width = int(conf.IMG_RES[0])
        cam_info_msg.K = self.K.flatten()
        cam_info_msg.D = np.zeros(4)
        cam_info_msg.R = np.zeros(9)
        cam_info_msg.distortion_model = "radial-tangential"
        cam_info_msg.P = np.zeros((3,4))
        cam_info_msg.P[:3,:3] = self.K
        cam_info_msg.P = cam_info_msg.P.flatten()
        self.cam_info_pub.publish(cam_info_msg)

    def publish_pose(self, j):
        # camera transform
        pos = self.T[:,3]
        self.T[:3,1:3] *= -1 # this hack is needed to convert from pybullet to ros/tf format.
        quat = tf.transformations.quaternion_from_matrix(self.T)
        self.br.sendTransform(pos, 
                              quat,
                              rospy.Time.now(),
                              f"drone_{j}/camera",
                              f"world")   
        campose_msg = PoseStamped()
        # campose_msg.header = Header()
        # pose_msg.header.stamp = rospy.Time.now()
        campose_msg.header.frame_id = "world"
        # campose_msg.pose = Pose(position=, orientation = )
        campose_msg.pose.position = Point(pos[0],pos[1],pos[2])
        campose_msg.pose.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
        self.sensor_pose_pubs[j].publish(campose_msg)

    def update_views(self, pos, quat, i):
        if conf.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(pos)
        self.cam_views[i] = p.computeViewMatrix(cameraEyePosition=pos + np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        self.cam_projs[i] =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L+0.1,
                                                      farVal=1000.0
                                                      )
        self.T = np.linalg.inv(np.array(self.cam_views[i]).reshape(4,4)).T

        f = self.cam_projs[i, 0]
        self.K = np.array([[f*conf.IMG_RES[0]/2, 0,                   conf.IMG_RES[0]/2],
                           [0,                   f*conf.IMG_RES[1]/2, conf.IMG_RES[1]/2],
                           [0,                   0,                   1]])

    def publish_boxes(seg):
        bb_max = np.max(np.argwhere(seg==4), axis=0)
        bb_min = np.min(np.argwhere(seg==4), axis=0)

    def publish_sem_cloud(self, pub, seg, dep):
        ds = 3
        seg = seg[::ds, ::ds]
        dep = dep[::ds, ::ds]

        seg_mask = np.argwhere((seg==4) * (dep!=0))

        K = self.K
        K[0:2,:] /= ds 

        pcl_msg = create_point_cloud(dep[seg_mask[:,0], seg_mask[:,1]], seg_mask, K)
        pub.publish(pcl_msg)

    def update_images(self, event=None):
        # SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        
        for i in range(self.num_drones):
            self.update_views(self.env.pos[i,:], self.env.quat[i,:], i)
            start = time.time()
            [w, h, rgb, dep, seg] = p.getCameraImage(width=conf.IMG_RES[0],
                                                    height=conf.IMG_RES[1],
                                                    shadow=1,
                                                    viewMatrix=tuple(self.cam_views[i]),
                                                    projectionMatrix=tuple(self.cam_projs[i]),
                                                    flags=p.ER_USE_PROJECTIVE_TEXTURE,
                                                    physicsClientId=self.CLIENT,
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL 
                                                    )
            # print(self.cam_views[i])                                                    
            rgb = np.reshape(rgb, (h, w, 4))
            dep = np.reshape(dep, (h, w))
            seg = np.reshape(seg, (h, w))

            # post process depth
            far = 1000
            near = self.L+0.1
            dep = far * near / (far - (far - near) * dep)
            dep[dep>5] =0

            # post process segmentation
            seg += 1 # because background is -1 and we turn it into 0


            self.publish_sem_cloud(self.pcl_pub, seg, dep)

            # print(np.unique(seg))
            # background, floor, pillars, target, drones
            # cmap = plt.get_cmap('tab10', 4+conf.DEFAULT_NUM_DRONES) # the number here will have to be set manualrly to fake unique colors for the segmentation data
        
            # # # Create a new image with unique colors for each segment
            # seg = cmap(seg, bytes=True) # bytes means uin8 from 0-255
            # print(seg.shape)
            
            # seg[seg_mask[:,0], seg_mask[:,1],:] = [255,0,0,255] 
            # print(seg[seg_mask[:,0], seg_mask[:,1],:])
            self.publish_img(self.dep_pubs[i], dep)
            # self.publish_img(self.seg_pubs[i], seg[...,:-1], "rgb8")
            
            self.publish_pose(i)
            self.publish_caminfo(i)
        # return rgb, dep, seg

if __name__=="__main__":
    sensors = Sensors()