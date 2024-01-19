from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import numpy as np
import pybullet as p

class Env(CtrlAviary):
    def __init__(self, drone_model: DroneModel = DroneModel.CF2X, num_drones: int = 1, neighbourhood_radius: float = np.inf, initial_xyzs=None, initial_rpys=None, physics: Physics = Physics.PYB, pyb_freq: int = 240, ctrl_freq: int = 240, gui=False, record=False, obstacles=False, user_debug_gui=False, output_folder='results'):
        super().__init__(drone_model, num_drones, neighbourhood_radius, initial_xyzs, initial_rpys, physics, pyb_freq, ctrl_freq, gui, record, obstacles, user_debug_gui, output_folder)

    def _addObstacles(self):
        """Remember to add separate urdfs for all the things you want to fake segmentation for"""
        # p.loadURDF(xacro2urdf("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_wall.xacro"),physicsClientId=self.CLIENT)
        # p.loadURDF(xacro2urdf("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_floor.xacro"),physicsClientId=self.CLIENT)
        # p.loadURDF(xacro2urdf("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_ceiling.xacro"),physicsClientId=self.CLIENT)
        # p.loadURDF("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_floor2.urdf",physicsClientId=self.CLIENT)
        # p.loadURDF("/root/gpdhydra_ws/src/gpd_ros/urdf/gallery_first_level_wall2.urdf",physicsClientId=self.CLIENT)
        # p.loadSDF("/root/racer_ws/src/RACER/gpd_ros/urdf/gazebo/world2.sdf",physicsClientId=self.CLIENT)
        # p.loadSDF("/root/model_editor_models/gazebo_walless/model.sdf", physicsClientId=self.CLIENT)
        # p.loadSDF("/root/model_editor_models/gazebo_walless/pillars.sdf", physicsClientId=self.CLIENT)
        # p.loadURDF("/root/racer_ws/src/RACER/gpd_ros/urdf/drone.urdf", physicsClientId=self.CLIENT)
        p.loadURDF("/root/racer_ws/src/RACER/gpd_ros/urdf/floor.urdf", physicsClientId=self.CLIENT)
        p.loadURDF("/root/racer_ws/src/RACER/gpd_ros/urdf/pillars.urdf", physicsClientId=self.CLIENT)
        p.loadURDF("/root/racer_ws/src/RACER/gpd_ros/urdf/target.urdf", physicsClientId=self.CLIENT)
