
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
import shutil
import hardware.oct_serial as oct_serial
import hardware.camera_api
import hardware.laser_pointer_control as laser_pointer_control

import exp_system_integration as  exp1_robot_id_oct_laser

"""global constant"""
# TODO: find an efficient way to label all these flags

"""axis calibration"""
FLAG_LASER_AXIX_ORIENTATION_CALIB = "false"

"""FID testing"""
FLAG_LASER_TUMORID_FID_TEST = "false"
FLAG_LASER_DIODE_FID_TEST = "false"

"""Inverse kinematics"""
FLAG_LASER_diode_IK = "false"

"""other functions"""
FLAG_SURFACE_SCAN = "false"

############################################

"""exvivo test"""
FLAG_EXVIVO_TISSUE_SCAN = "false"

FLAG_EXVIVO_TISSUE_SCAN_WITH_DIODE = "true"

"""speed test"""
FLAG_LASER_FIBER_SPEED_TEST = "false"

"""mice test"""
FLAG_LASER_FIBER_MICE_EXP = "false"

"""fiber calibration"""
FLAG_LASER_FIBER_CALIB = "false"

"""orientation calibration"""
# tumorid
FLAG_LASER_INCIDENT_ORIENTATION_CALIB = "false"
# diode 
FLAG_DIODE_INCIDENT_ORIENTATION_CALIB = "false"

"""traj test"""
# diode
FLAG_DIODE_TRAJ_TEST = "false"
# tumorid 
FLAG_TUMORID_TRAJ_TEST = "false"

class exp_system_test(): 

    def __init__(self):
        """initialize the camera and the robot-stage modules"""
        self.exp1_unit = exp1_robot_id_oct_laser.exp_unit_test() 

    def mice_exp_planar_raster_scan(self, traj_track = [], path_main_ref = []):

        """setpoint testing for the experiments (mice experiments)
        1. move the robot to the home configuration
        2. move the robot the first setpoint configuration
        3. move the robot following the pre-defined scanning trajectory."""
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = path_main_ref

        # mode of index position fixed
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false" 

        # mode of laser and off mode
        para_dict["mode_laser_on_and_off"] = "true"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "true"

        # tumorid setting
        para_dict["mode_id"] = "true"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
    
        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # surface-oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001

        # home position with the robot initialization 
        time_to_home = 6.0 
        para_dict["q_home"] = [0.0, -4.86, -0.954, 1.618, -0.36, 0.101, -4.172, 0.0]  

        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 6.0
        para_dict["q_setpoint_1"] = [0.0, -4.875, -0.824, 1.615, -0.499, 0.101, -4.172, 0.0]

        # scanning region initialization 
        # para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        time_to_setpoint_2 = 6.0
        para_dict["q_setpoint_2"] = [0.0, -4.875, -0.794, 1.618, -0.532, 0.101, -4.232, 0.0]

        # setup the robot configuration.
        # traj_track = []
        traj_use = np.load(traj_track)
        para_dict["time_to_first_pos"] = 6.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 

        # important settings of the configuration 
        para_dict["time_stop_each_iter"] = 0.5
        para_dict["integration_time"] = 1.0
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def mice_exp_label_color(self, traj_track = [], mode_of_label = "true", path_main_ref = []): 
        """scan and punch workflow
        1. follow a single line """
        
        # main folder
        # parameter input with a dict-format. 
        para_dict = {}
        para_dict["path_main"] = path_main_ref

        # mode of index position fixed
        para_dict["idx_stop"] = 0
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        if mode_of_label == "true": 
            para_dict["mode_punch_biopsy"] = "true"
        else:
            para_dict["mode_punch_biopsy"] = "false"

        # surface-oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # mode of laser 
        # mode-1: on-and-off
        # mode-2: on-all
        para_dict["mode_laser_on_and_off"] = "true"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "true"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # tumorid setting
        para_dict["mode_id"] = "true"
        # para_dict["time_stop_each_iter"] = 1.0
        # para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001

        # home position with the robot initialization 
        time_to_home = 5.0 
        para_dict["q_home"] = [0.0, -4.86, -0.954, 1.618, -0.36, 0.101, -4.172, 0.0]

        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 5.0
        para_dict["q_setpoint_1"] = [0.0, -4.875, -0.824, 1.615, -0.499, 0.101, -4.172, 0.0]
                                    
        # move from the scanning region to the setpoint-2 
        time_to_setpoint_2 = 6.0
        para_dict["q_setpoint_2"] = [0.0, -4.875, -0.794, 1.618, -0.532, 0.101, -4.232, 0.0]

        # setup the robot configuration.
        traj_use = np.load(traj_track)
        para_dict["time_to_first_pos"] = 5.0
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0       # TODO: the first point (to be determined)

        # important settings of the configuration 
        para_dict["time_stop_each_iter"] = 0.5
        para_dict["integration_time"] = 1.0
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def speed_roi_test_only(self): 
        """raster scanning under the oct"""
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 12.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 12.0
        para_dict["q_setpoint_1"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # scanning region initialization 
        # para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # speed test
        traj_use = np.load("./data_calib/fiber_speed_test_traj.npy")

        para_dict["time_stop_each_iter"] = 0.025                     # 0.025 (very efficient scanning)     0.1, 0.05, 0.025
        para_dict["time_to_first_pos"] = 12.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def tumorid_fid_test(self): 
        """raster scanning under the oct"""
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = []
        # for mice study only
        flag_check_fiber_ik = input("check the fiber ik update file (yes or no)")
        if flag_check_fiber_ik == "yes":
            path_laser_use = "./database/data_ik/traj_ik_test_tumorid_phantom.npy"
        else:
            exit()

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "true"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.0001

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.0001

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 10.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 10.0
        para_dict["q_setpoint_1"] = [0.0, -4.437, -0.868, 1.739, -1.117, 0.099, -4.341, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.437, -0.868, 1.739, -1.117, 0.099, -4.341, 0.0]

        # important settings
        traj_use = np.load( path_laser_use )

        para_dict["time_stop_each_iter"] = 0.5                     # 0.025 (very efficient scanning)     0.1, 0.05, 0.025
        para_dict["time_to_first_pos"] = 12.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 120.0 / 1000
        para_dict["integration_time"] = 1.0
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def mice_fid_test(self): 
        """raster scanning under the oct"""
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = []
        
        # for mice study only
        flag_check_fiber_ik = input("check the fiber ik update file (yes or no)")
        if flag_check_fiber_ik == "yes":
            path_laser_use = "./database/data_ik/traj_ik_test_fiber.npy"
        else:
            exit()

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.006

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.006

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 12.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 12.0
        para_dict["q_setpoint_1"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # scanning region initialization 
        # para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # important settings
        traj_use = np.load( path_laser_use )

        para_dict["time_stop_each_iter"] = 0.5                     # 0.025 (very efficient scanning)     0.1, 0.05, 0.025
        para_dict["time_to_first_pos"] = 12.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def mice_exp_surface_raster_diode_traj_test(self): 
        """raster scanning under the oct"""
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = []
        # for mice study only
        flag_check_fiber_ik = input("check the fiber ik update file (yes or no)")
        if flag_check_fiber_ik == "yes":
            path_laser_use = "./database/data_ik/traj_ik_test_fiber.npy"
        else:
            exit()

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 12.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 12.0
        para_dict["q_setpoint_1"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # scanning region initialization 
        # para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # important settings
        traj_use = np.load( path_laser_use )

        para_dict["time_stop_each_iter"] = 0.025                     # 0.025 (very efficient scanning)     0.1, 0.05, 0.025
        para_dict["time_to_first_pos"] = 12.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def mice_exp_surface_raster_scan_laser(self): 
        """raster scanning under the oct"""
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"
        # for mice study only
        flag_check_fiber_ik = input("check the fiber ik update file (yes or no)")
        if flag_check_fiber_ik == "yes":
            path_laser_use = "./database/data_ik/traj_ik_test_fiber.npy"
        else:
            exit()

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 12.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 12.0
        para_dict["q_setpoint_1"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # scanning region initialization 
        # para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # important settings
        traj_use = np.load( path_laser_use )

        para_dict["time_stop_each_iter"] = 0.025                     # 0.025 (very efficient scanning)     0.1, 0.05, 0.025
        para_dict["time_to_first_pos"] = 12.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def calib_cam_intrinsic(self):
        """both left and right images"""
        
        # main folder + basic parameters 
        path_img_base = []
        path_img_left = path_img_base + r"left_cam\\"
        path_img_right = path_img_base + r"right_cam\\"
        mode_cam_left = "true"
        mode_cam_right = "true"
        exposure_time_of_ref_img_left = 0.005
        exposure_time_of_laser_spot_img_left = 0.005
        exposure_time_of_ref_img_right = 0.005
        exposure_time_of_laser_spot_img_right = 0.005
        num_of_img = 30

        # define the image object 
        # left camera
        if mode_cam_left == "true":
            print("left camera initialization")
            input("left image (remove oct)")
            para_dict_left = {} 
            para_dict_left["exposure_time_of_ref_img_left"] = exposure_time_of_ref_img_left
            para_dict_left["exposure_time_of_laser_spot_img_left"] = exposure_time_of_laser_spot_img_left
            para_dict_left["path_main"] = path_img_base 
            self.exp1_unit.cam_left_init(para_dict_left=para_dict_left) 
        # right camera 
        if mode_cam_right == "true":
            print("right camera initialization")
            input("right image (remove oct)")
            para_dict_right = {} 
            para_dict_right["exposure_time_of_ref_img_right"] = exposure_time_of_ref_img_right
            para_dict_right["exposure_time_of_laser_spot_img_right"] = exposure_time_of_laser_spot_img_right
            para_dict_right["path_main"] = path_img_base 
            self.exp1_unit.cam_right_init(para_dict_right=para_dict_right) 

        # loop to take the images
        # loop through all the images 
        for idx_img in range(num_of_img):
            print("idx_img = ", idx_img)
            input("press to collect the images")
            if mode_cam_left == "true":
                img_tmp = self.exp1_unit.camera_obj_left.get_one_image()
                cv2.imwrite( path_img_left + str(idx_img) + "_left.jpg", img_tmp)
            if mode_cam_right == "true": 
                img_tmp = self.exp1_unit.camera_obj_right.get_one_image()
                cv2.imwrite( path_img_right + str(idx_img) + "_right.jpg", img_tmp)
        print("finished the camera intrinsic images")

        return 0

    def laser_cutting_fid_test(self):
        """cutting laser calibration of the laser orientation (as a raster scanning mode)"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "true"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.417, -0.835, 1.575, -0.732, 0.098, -4.241, 0.0]  # [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load("./data_calib/laser_cutting_fid_test.npy")    
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 120.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def laser_fiber_incident_orientation_calibration_traj(self):
        """fiber testing only"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]  # [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # important settings
        traj_use = np.load("./data_calib/laser_fiber_incident_dir_raster.npy")    
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 120.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def laser_diode_incident_orientation_calibration_traj(self):
        """cutting laser calibration of the laser orientation (as a raster scanning mode)"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "true"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.417, -0.835, 1.575, -0.732, 0.098, -4.241, 0.0]  # [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load("./data_calib/laser_diode_incident_dir_raster.npy")    
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 120.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def laser_tumorid_roi_scan(self):
        """calibration of the laser orientation (as a raster scanning mode)"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "true"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00005

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00005

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.417, -0.835, 1.575, -0.732, 0.098, -4.241, 0.0]  # [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load("./data_calib/laser_tumorid_roi_scan.npy")
        para_dict["time_stop_each_iter"] = 0.25
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 120.0 / 1000
        para_dict["integration_time"] = 0.25
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def laser_incident_orientation_calibration_traj(self):
        """calibration of the laser orientation (as a raster scanning mode)"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "true"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.0005

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.0005

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.417, -0.835, 1.575, -0.732, 0.098, -4.241, 0.0]  # [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.429, -0.799, 1.539, -1.038, 0.092, -4.272, 0.0]

        # important settings
        traj_use = np.load("./data_calib/laser_incident_dir_raster.npy")
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 120.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def laser_cutting_axis_orientation_calibration_traj(self):
        """calibration of the laser orientation (as a raster scanning mode)"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "true"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        # this is adjusted  
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.394, -0.953, 1.776, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.394, -0.953, 1.776, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load("./data_calib/laser_cutting_axis_orientation_raster_3.npy")
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 90.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def laser_axis_orientation_calibration_traj(self):
        """calibration of the laser orientation (as a raster scanning mode)"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_calib/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        # this is adjusted  
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.394, -0.953, 1.776, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.394, -0.953, 1.776, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load("./data_calib/laser_axis_orientation_raster.npy")
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 90.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def dual_laser_roi_test(self): 
        """test the feasibility of the dual-laser configuration"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_dual_laser_roi_test/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load("./data_dual_laser_roi_test/traj_ik_test.npy")
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 75.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def ik_tumorid_laser_1(self): 
        """test the ik with the trajectory
        {tumorid} reference frame 
        """
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_ik/"

        # reference name for the file 
        flag_laser = "tumorid"
        path_traj_file_tmp = "./data_ik/traj_ik_test_" + flag_laser + ".npy"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "true"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001 # 0.00008 #

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001 # 0.00008 #

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.417, -0.835, 1.575, -0.732, 0.098, -4.241, 0.0]  # [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load( path_traj_file_tmp )
        para_dict["time_stop_each_iter"] = 0.25
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 95.0 / 1000
        para_dict["integration_time"] = 0.05
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def ik_diode_laser_exvivo(self): 
        """same code as the mice cutting study"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_exvivo_official/post_scan_diode/"
        # for exvivo study ~= mice study
        path_laser_use = "./database/data_ik/traj_ik_test_fiber.npy"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.0001    # 0.00005

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.0001

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 12.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 12.0
        para_dict["q_setpoint_1"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # scanning region initialization 
        # para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # important settings
        traj_use = np.load( path_laser_use )
        print("traj_use = ", traj_use.shape)
        # input("check")

        para_dict["time_stop_each_iter"] = 0.10                     # 0.025 (very efficient scanning) -> return the results + cases.
        para_dict["time_to_first_pos"] = 12.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def ik_fiber_laser_exvivo(self): 
        """same code as the mice cutting study"""

        # main folder
        para_dict = {}
        para_dict["path_main"]  = "./data_calib/"
        path_laser_use          = "./data_ik/traj_ik_test_fiber.npy"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 12.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 12.0
        para_dict["q_setpoint_1"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # scanning region initialization 
        # para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.332, -0.841, 1.512, -0.92, 0.102, -4.268, 0.0]

        # important settings
        traj_use = np.load( path_laser_use )
        print("traj_use = ", traj_use.shape)
        # input("check")

        para_dict["time_stop_each_iter"] = 0.050                     # 0.025 (very efficient scanning) -> return the results + cases.
        para_dict["time_to_first_pos"] = 12.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def ik_diode_laser_2(self): 
        """test the ik with the trajectory
        {diode} reference frame 
        """
        
        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_ik/"

        # reference name for the file 
        flag_laser = "diode"
        path_traj_file_tmp = "./data_ik/traj_ik_test_" + flag_laser + ".npy"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "true"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008 # 0.0005 # 

        # right camera
        para_dict["mode_cam_right"] = "true"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008  #  0.0005 # 

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.417, -0.835, 1.575, -0.732, 0.098, -4.241, 0.0]  # [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load( path_traj_file_tmp )
        para_dict["time_stop_each_iter"] = 0.25
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 75.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def ik_test(self): 
        """test the ik with the trajectory"""

        # main folder
        para_dict = {}
        para_dict["path_main"] = "./data_ik/"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 8.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 8.0
        para_dict["q_setpoint_1"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.416, -0.97, 1.99, -1.002, 0.098, -4.346, 0.0]

        # important settings
        traj_use = np.load("./data_ik/traj_ik_test.npy")
        para_dict["time_stop_each_iter"] = 1.0 
        para_dict["time_to_first_pos"] = 10.0 
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 75.0 / 1000
        para_dict["integration_time"] = 0.1
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def calib_oct_to_cam(self):
        """camera-to-oct registration
        1. camera images.
        2. oct fiducial detection based on the C-scans.
        """

        # main folder + basic parameters 
        path_img_base   = []
        path_img_left   = []
        path_img_right  = []
        mode_cam_left = "true"
        mode_cam_right = "true"
        mode_oct = "true"
        exposure_time_of_ref_img_left = 0.005
        exposure_time_of_laser_spot_img_left = 0.005
        exposure_time_of_ref_img_right = 0.005
        exposure_time_of_laser_spot_img_right = 0.005
        num_of_pose = 6

        # define the image object 
        # left camera
        if mode_cam_left == "true":
            print("left camera initialization")
            input("left image (remove oct)")
            para_dict_left = {} 
            para_dict_left["exposure_time_of_ref_img_left"] = exposure_time_of_ref_img_left
            para_dict_left["exposure_time_of_laser_spot_img_left"] = exposure_time_of_laser_spot_img_left
            para_dict_left["path_main"] = path_img_base 
            self.exp1_unit.cam_left_init(para_dict_left=para_dict_left) 
        # right camera 
        if mode_cam_right == "true":
            print("right camera initialization")
            input("right image (remove oct)")
            para_dict_right = {} 
            para_dict_right["exposure_time_of_ref_img_right"] = exposure_time_of_ref_img_right
            para_dict_right["exposure_time_of_laser_spot_img_right"] = exposure_time_of_laser_spot_img_right
            para_dict_right["path_main"] = path_img_base 
            self.exp1_unit.cam_right_init(para_dict_right=para_dict_right) 

        # loop to take the images
        # loop through all the images 
        for idx_pose in range(num_of_pose):
            print("pose = ", idx_pose)
            input("get the cover for the oct")
            input("press to collect the images")
            if mode_cam_left == "true":
                img_tmp = self.exp1_unit.camera_obj_left.get_one_image()
                cv2.imwrite( path_img_left + str(idx_pose + 1) + "_left.jpg", img_tmp)
            if mode_cam_right == "true": 
                img_tmp = self.exp1_unit.camera_obj_right.get_one_image()
                cv2.imwrite( path_img_right + str(idx_pose + 1) + "_right.jpg", img_tmp)
            input("remove the cover and collect oct")
            if mode_oct == "true":
                print("oct serial initalization")
                input("move the cover and start oct scan")
                file_name_oct_serial = str(idx_pose + 1) + "_oct"
                oct_token = self.exp1_unit.oct.octvolscan(file_name_oct_serial)
                if oct_token == False:
                    print("oct scan failed")
                    return 0
        print("finished the camera intrinsic images")

    def exvivo_exp_surface_raster_scan(self, path_ref = [], path_traj_use = []): 
        """raster scanning under the oct"""
        
        # automated folder problem

        # important settings
        para_dict = {}
        
        # para_dict["path_main"] = path_ref
        para_dict["path_main"] = path_ref
        path_of_traj = path_traj_use 

        # setpoint study of the ex-vivo tissue data
        traj_use = np.load(path_of_traj)
        para_dict["time_stop_each_iter"] = 0.5
        para_dict["time_to_first_pos"] = 10.0    
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0

        # channel for tumorid
        para_dict["mode_id"] = "true"

        # camera
        para_dict["mode_cam_left"] = "true"
        para_dict["mode_cam_right"] = "true"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "true"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00006

        # right camera
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00006

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 10.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 10.0
        para_dict["q_setpoint_1"] = [0.0, -4.437, -0.868, 1.739, -1.117, 0.099, -4.341, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.437, -0.868, 1.739, -1.117, 0.099, -4.341, 0.0]

        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def mice_exp_surface_raster_scan(self, path_ref = []): 
        """raster scanning under the oct"""
        
        # automated folder problem

        # important settings
        para_dict = {}
        # para_dict["path_main"] = path_ref
        para_dict["path_main"] = path_ref
        path_of_traj = para_dict["path_main"] + "surface_raster_scan_8.npy"      
        # setpoint
        traj_use = np.load(path_of_traj)
        para_dict["time_stop_each_iter"] = 1.0
        para_dict["time_to_first_pos"] = 8.0    
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0 
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["integration_time"] = 1.0

        # channel for tumorid
        para_dict["mode_id"] = "true"

        # camera
        para_dict["mode_cam_left"] = "true"
        para_dict["mode_cam_right"] = "true"

        # mode of laser on and off 
        para_dict["mode_laser_on_and_off"] = "true"

        # mode of index position fixed at the particular position 
        para_dict["idx_stop"] = 0 
        para_dict["mode_fix_index_pos"] = "false"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # tumorid setting
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # left camera
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.00008

        # right camera
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.00008

        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # home position with the robot initialization 
        time_to_home = 10.0
        para_dict["q_home"] = [0.0, -4.417, -1.126, 1.99, -0.732, 0.098, -4.241, 0.0]
                              
        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 10.0
        para_dict["q_setpoint_1"] = [0.0, -4.437, -0.868, 1.739, -1.117, 0.099, -4.341, 0.0]

        # scanning region initialization 
        para_dict["q_roi_init"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.35, 0.0] 

        # move from the scanning region to the setpoint-2 
        para_dict["q_setpoint_2"] = [0.0, -4.437, -0.868, 1.739, -1.117, 0.099, -4.341, 0.0]

        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

    def scan_fix_orientation_scan(self, idx_angle = "90", traj_track = [], idx_stop = [], path_main_ref = []):
        """move the robot to the fixed distance"""
        
        # main folder
        # parameter input with a dict-format. 
        para_dict = {}
        para_dict["path_main"] = path_main_ref

        # mode of index position fixed
        para_dict["idx_stop"] = idx_stop
        para_dict["mode_fix_index_pos"] = "true"

        # punch biopsy mode 
        para_dict["mode_punch_biopsy"] = "false" 

        # surface-oct scanning mode
        para_dict["mode_surface_scan_under_oct"] = "false"

        # mode of laser 
        # mode-1: on-and-off
        # mode-2: on-all
        para_dict["mode_laser_on_and_off"] = "false"

        # vimba camera setting -> color camera image collection. 
        para_dict["mode_vimba_camera"] = "false"

        # stage setting
        para_dict["mode_stage"] = "false"

        # rgbd-color camera
        para_dict["mode_rgbd_color_cam"] = "false"
        para_dict["cam_rgbd_color_exposure"] = 100.00
        para_dict["path_color_img"] = para_dict["path_main"] + "tumorid_img_color/"

        # tumorid setting
        para_dict["mode_id"] = "false"
        para_dict["time_stop_each_iter"] = 1.0
        # para_dict["laser_current_id"] = 120.0 / 1000
        para_dict["path_tumorid_data"] = para_dict["path_main"] + "tumorid_spectrum/"
        para_dict["path_tumorid_img_left"] = para_dict["path_main"] + "tumorid_img_left/"
        para_dict["path_tumorid_img_right"] = para_dict["path_main"] + "tumorid_img_right/"
        para_dict["mode_scangrid_pattern"] = "connect"
        
        # robot setting
        para_dict["mode_robot"] = "true"
        para_dict["x_init_pos_stage"] = 185
        para_dict["x_final_pos_stage"] = 195
        para_dict["y_init_pos_stage"] = 70
        para_dict["y_final_pos_stage"] = 80
        para_dict["x_step_size"] = 2
        para_dict["y_step_size"] = 2 
        para_dict["x_max_stage"] = 250
        para_dict["y_max_stage"] = 150

        # oct setting
        para_dict["mode_oct"] = "false"
        para_dict["file_name_oct_serial"] = "system_test"

        # simulation setting
        para_dict["mode_sim"] = "false"

        # laser pointer setting
        para_dict["mode_laser_green"] = "false"
        para_dict["mode_laser_red"] = "false"

        # left camera
        para_dict["mode_cam_left"] = "false"
        para_dict["exposure_time_of_ref_img_left"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_left"] = 0.001

        # right camera
        para_dict["mode_cam_right"] = "false"
        para_dict["exposure_time_of_ref_img_right"] = 0.005
        para_dict["exposure_time_of_laser_spot_img_right"] = 0.001

        # home position with the robot initialization 
        time_to_home = 6.0 
        para_dict["q_home"] = [0.0, -4.86, -0.954, 1.618, -0.36, 0.101, -4.172, 0.0]

        # move from the home to the first setpoint-1
        time_to_setpoint_1 = 6.0
        para_dict["q_setpoint_1"] = [0.0, -4.875, -0.824, 1.615, -0.499, 0.101, -4.172, 0.0]
                                    
        # move from the scanning region to the setpoint-2 
        time_to_setpoint_2 = 6.0
        para_dict["q_setpoint_2"] = [0.0, -4.875, -0.794, 1.618, -0.532, 0.101, -4.232, 0.0]

        # setup the robot configuration.
        traj_use = np.load(traj_track)
        para_dict["time_to_first_pos"] = 6.5
        para_dict["traj_robot_config"] = traj_use
        para_dict["idx_align_center"] = 0       

        # important settings of the configuration 
        para_dict["time_stop_each_iter"] = 0.5
        para_dict["integration_time"] = 1.0
        # para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["laser_current_id"] = 220.0 / 1000
        para_dict["time_to_home"] = time_to_home 
        para_dict["time_to_setpoint_1"] = time_to_setpoint_1

        # test the system pipeline
        self.exp1_unit.raster_scan_core(para_dict=para_dict) 

class framework_calibration():
    
    def __init__(self): 
        self.class_utility_test = exp_system_test()
        
        # unique flags 
        # self.flag_laser_incidence_orientation_calib = "false"

    def camera_intrinsic_calib(self):
        """camera intrinsic calibrations
        1. collect images from both camera -> {left} + {right} 
        2. different poses (height + orientations)
        3. manual copy to ensure the correctness of the workflow 
        """
        self.class_utility_test.calib_cam_intrinsic()
        # num_of_imgs = 10 
        # for idx_img in range(num_of_imgs): 
        #     input("move the image to the target position")
        #     self.class_utility_test.calib_cam_intrinsic()
        #     input("save the image to the target folder")

    def camera_to_oct_calib(self):

        self.class_utility_test.calib_oct_to_cam() 

    def laser_axis_orientation_calib(self):
        """fiducial as the tool-tip
        raster scanning along a 10 x 10 meshgrids""" 
        self.class_utility_test.laser_axis_orientation_calibration_traj()

    def laser_cutting_axis_orientation_calib(self):

        if FLAG_LASER_AXIX_ORIENTATION_CALIB == "true":
            self.class_utility_test.laser_cutting_axis_orientation_calibration_traj()

    def laser_incidence_orientation_calib(self):
        """important""" 
        # if FLAG_LASER_INCIDENT_ORIENTATION_CALIB == "true": 
        self.class_utility_test.laser_incident_orientation_calibration_traj()

    def laser_fiber_incidence_orientation_calib(self):
        
        if FLAG_LASER_FIBER_CALIB == "true": 
            self.class_utility_test.laser_fiber_incident_orientation_calibration_traj()
        else:
            print("please set the global variable -> exit()")
            exit()

    def laser_cutting_fid_test(self): 
        """diode fid test + average filtering testing"""
        self.class_utility_test.laser_cutting_fid_test()

    def laser_tumorid_fid_test(self): 
        """tumorid fid test + average filtering testing"""
        self.class_utility_test.laser_tumorid_fid_test()

class framework_experiment(): 
    
    """generate the experimental maps"""
    def __init__(self): 
        self.class_utility_test = exp_system_test()
        # self.class_utility_calib = framework_calibration() 

    def ik_tumorid(self):
        """test the ik based on the tumorid calibration"""
        self.class_utility_test.ik_tumorid_laser_1()

    def ik_diode(self): 
        """test the ik based on the tumorid calibration"""
        self.class_utility_test.ik_diode_laser_2()

    def ik_fiber(self): 
        """test the ik based on the tumorid calibration"""
        if FLAG_EXVIVO_TISSUE_SCAN == "true":
            self.class_utility_test.ik_fiber_laser_exvivo()
        else:
            print("check")
            exit() 

    def ik_laser_exvivo_with_fiber(self): 
        """test the ik based on the tumorid calibration"""
        if FLAG_EXVIVO_TISSUE_SCAN == "true":
            self.class_utility_test.ik_fiber_laser_exvivo()
        else:
            print("check global")
            exit() 

    def ik_laser_exvivo_with_diode(self): 
        if FLAG_EXVIVO_TISSUE_SCAN_WITH_DIODE == "true":
            self.class_utility_test.ik_diode_laser_exvivo()

class exp_main(): 

    def __init__(self):

        self._class_framework_test  = framework_experiment() 
        self._class_framework_calib = framework_calibration() 
        self.class_utility_test     = exp_system_test()

    def ik_exp(self):
        """inverse kinematics tesing"""
        
        # tumorid: laser-1
        self._class_framework_test.ik_tumorid()

        # diode: laser-2
        self._class_framework_test.ik_diode() 

        # fiber: laser-3: 
        self._class_framework_test.ik_fiber()

    def calib_exp(self): 
        """system calibration testing"""

        # calib-1: oct-to-camera
        self._class_framework_calib.camera_to_oct_calib()

        # calib-2: oct-to-laser-axis
        self._class_framework_calib.laser_cutting_axis_orientation_calib()             

        # calib-3: oct-to-laser-dir
        self._class_framework_calib.laser_fiber_incidence_orientation_calib()          

    def system_exp(self): 
        """system expeirments"""

        # fid testing
        if FLAG_LASER_FIBER_CALIB == "true":
            input("mice post scan")
            self.class_utility_test.mice_fid_test()
        else:
            print("check global variable")
            exit()

        # trajectory testing
        if FLAG_DIODE_TRAJ_TEST == "true":
            print("mice_s_test")
            self.class_utility_test.mice_exp_surface_raster_diode_traj_test()
        else:
            print("check global variable")
            exit()
            
        # roi testing 
        if FLAG_DIODE_TRAJ_TEST == "true":
            print("mice_circle_test")
            self.class_utility_test.mice_exp_surface_raster_diode_traj_test()
        else:
            print("check global variable")
            exit()

    def exvivo_exp(self): 
        """exvivo tissue experiments"""

        # step-1: pre-scan with tissue model
        path_ref      = []  # user define  
        path_traj_use = []  # user define 
        self._class_framework_test.class_utility_test.exvivo_exp_surface_raster_scan(path_ref=path_ref, path_traj_use=path_traj_use)
        
        # step-2a: laser-fiber
        input("exvivo cutting with diode")
        self._class_framework_test.ik_laser_exvivo_with_diode()

        # step-2b: laser-fiber 
        input("exvivo cutting the fiber")
        self._class_framework_test.ik_laser_exvivo_with_fiber()

    def mice_exp(self):  
        """official mice study"""

        # scan
        path_ref = [] # user define
        self.class_utility_test.mice_exp_surface_raster_scan(path_ref=path_ref)

        # resection
        if FLAG_LASER_FIBER_MICE_EXP == "true":
            print("mice exp test")
            self.class_utility_test.mice_exp_surface_raster_scan_laser()
        else:
            print("check global variable")
            exit()

    def mice_data_collect(self): 

        """mice data collection (raster scan)"""
        # surface scan 
        path_ref = []
        self.class_utility_test.mice_exp_surface_raster_scan(path_ref=path_ref)

        # tumor resection experiment data collection
        idx_angle       = 90        # for example, 90-degree (perpendicular to the tissue)
        idx_stop        = 35 + 1    # for example, stop at the index-position to check for correctness.
        traj_track      = [] # path to raster scanning pre-defined trajectory 
        path_main_ref   = [] # path for saving data 
        
        # setpoint at the middle center
        self.class_utility_test.scan_fix_orientation_scan(idx_angle=idx_angle, traj_track=traj_track, idx_stop = idx_stop, path_main_ref = path_main_ref)
        
        # planar raster scanning
        self.class_utility_test.mice_exp_planar_raster_scan(traj_track = traj_track, path_main_ref = path_main_ref) 

    def mice_histopathology_label(self): 

        """histopathology: line-scan"""
        # move to the five point scanning centers. 
        idx_angle       = 90 
        idx_stop        = 2 + 1
        traj_track      = [] # labeling scanning trajectory
        path_main_ref   = [] # path to save data 
        self.class_utility_test.scan_fix_orientation_scan(idx_angle=idx_angle, traj_track=traj_track, idx_stop=idx_stop, path_main_ref=path_main_ref)

        # first pass: tumorid scan (no label)
        traj_track      = [] # labeling scanning trajectory
        path_main_ref   = [] # path to save data 
        mode_of_label   = "false"
        self.class_utility_test.mice_exp_label_color(traj_track=traj_track, mode_of_label=mode_of_label, path_main_ref=path_main_ref) 

        # second pass: tumorid scan (with labeling procedure)
        traj_track      = [] # labeling scanning trajectory
        path_main_ref   = [] # path to save data 
        mode_of_label   = "true"
        self.class_utility_test.mice_exp_label_color(traj_track=traj_track, mode_of_label=mode_of_label, path_main_ref=path_main_ref) 

if __name__ == "__main__":

    """testing"""
    exp_test_class = exp_main()

    """ik testing"""
    exp_test_class.ik_exp()

    """system calibration"""
    exp_test_class.calib_exp()

    """system testing"""
    exp_test_class.system_exp()

    """exvivo testing"""
    exp_test_class.exvivo_exp()

    """mice testing"""
    exp_test_class.mice_exp()

    """mice data collection"""
    exp_test_class.mice_data_collect()

    """histopathology labeling"""
    exp_test_class.mice_histopathology_label()
