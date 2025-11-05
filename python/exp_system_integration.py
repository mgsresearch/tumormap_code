
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

import hardware.oct_serial as oct_serial
import hardware.camera_api as camera_api
import hardware.laser_pointer_control as laser_pointer_control
import hardware.tumorid as tumorid
import hardware.robot_arm as robot_arm
import hardware.rgbd_camera as cam_rgbd

from pymba import *

# local modules
import exp_workflow as exp2_raster_scan_vision

"""important global flag definitions"""
FLAG_LASER_INCIDENT_DIR_CALIB = exp2_raster_scan_vision.FLAG_LASER_INCIDENT_ORIENTATION_CALIB
FLAG_LASER_AXIX_ORIENTATION_CALIB = exp2_raster_scan_vision.FLAG_LASER_AXIX_ORIENTATION_CALIB
FLAG_LASER_diode_IK = exp2_raster_scan_vision.FLAG_LASER_diode_IK
FLAG_LASER_TUMORID_FID_TEST = exp2_raster_scan_vision.FLAG_LASER_TUMORID_FID_TEST
FLAG_LASER_DIODE_FID_TEST = exp2_raster_scan_vision.FLAG_LASER_DIODE_FID_TEST
FLAG_LASER_FIBER_CALIB = exp2_raster_scan_vision.FLAG_LASER_FIBER_CALIB
FLAG_LASER_FIBER_SPEED_TEST = exp2_raster_scan_vision.FLAG_LASER_FIBER_SPEED_TEST
FLAG_SURFACE_SCAN = exp2_raster_scan_vision.FLAG_SURFACE_SCAN
FLAG_LASER_FIBER_MICE_EXP = exp2_raster_scan_vision.FLAG_LASER_FIBER_MICE_EXP
FLAG_EXVIVO_TISSUE_SCAN = exp2_raster_scan_vision.FLAG_EXVIVO_TISSUE_SCAN
FLAG_DIODE_INCIDENT_DIR_CALIB = exp2_raster_scan_vision.FLAG_DIODE_INCIDENT_ORIENTATION_CALIB
FLAG_DIODE_TRAJ_TEST = exp2_raster_scan_vision.FLAG_DIODE_TRAJ_TEST
FLAG_TUMORID_TRAJ_TEST = exp2_raster_scan_vision.FLAG_TUMORID_TRAJ_TEST
FLAG_EXVIVO_TISSUE_SCAN_WITH_DIODE = exp2_raster_scan_vision.FLAG_EXVIVO_TISSUE_SCAN_WITH_DIODE

class exp_unit_test():

    def __init__(self):
        """initialize the camera and the robot-stage modules"""

        # # camera
        self.camera_obj_left = camera_api.library_camera(cam_id="DFK 33UP1300 1")
        self.camera_obj_right = camera_api.library_camera(cam_id="DFK 33UP1300")

        # vimba camera setting 
        # self.cam_vimba = cam_vimba.StereoVimba() 

        # rgbd color camera
        # self.cam_rgbd_color = cam_rgbd.rgbd_module(flag_device="true")

        # oct
        self.oct = oct_serial.octserial()

        # tumorid
        flag_id = input("start the tumorid? (yes or no)")
        if flag_id == "yes":
            self.tumorid = tumorid.TumorID()

        # robot stage
        # self.stage = robot_stage.controlstage()

        # TODO: use the ur5-robot as a reference value 
        # set the hardware mode based on the robot mode
        # move_to: move to the next point of a 3d trajectory (in a 3d-line)
        self.robot = robot_arm.robot_ur5()
     
        # laser pointer
        # self.laser_diode = laser_pointer_control.pointercontrol() 

    def cam_left_init(self, para_dict_left):

        # setting
        exposure_time_of_ref_img_left = para_dict_left["exposure_time_of_ref_img_left"] 
        exposure_time_of_laser_spot_img_left = para_dict_left["exposure_time_of_laser_spot_img_left"] 
        path_main = para_dict_left["path_main"] 

        if (self.camera_obj_left.cam_obj.IC_IsDevValid(self.camera_obj_left.hGrabber_obj)):
            self.camera_obj_left.cam_obj.IC_StartLive(self.camera_obj_left.hGrabber_obj, 1)
        exposure_time_of_ref_img_left = para_dict_left["exposure_time_of_ref_img_left"]
        exposure_time_of_laser_spot_img_left = para_dict_left["exposure_time_of_laser_spot_img_left"]
        self.camera_obj_left.set_exposure_time(exposure_time_of_ref_img_left)
        time.sleep(0.5)
        img_ref_left = self.camera_obj_left.get_one_image()
        self.camera_obj_left.set_exposure_time(input_exposure_time=exposure_time_of_laser_spot_img_left)
        img_of_low_exposure_left = self.camera_obj_left.get_one_image()
        
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(img_ref_left)
        plt.subplot(1, 2, 2)
        plt.imshow(img_of_low_exposure_left)
        plt.show()

        flag_save_base_img = input("save the reference image (yes or no)?")
        if flag_save_base_img == "yes":
            path_img_reference = path_main + "img_base_left.jpg"
            path_img_low_exposure = path_main + "img_low_exposure_left.jpg"
            cv2.imwrite(path_img_reference, img_ref_left)
            cv2.imwrite(path_img_low_exposure, img_of_low_exposure_left)

    def cam_right_init(self, para_dict_right):

        # setting
        exposure_time_of_ref_img_right = para_dict_right["exposure_time_of_ref_img_right"] 
        exposure_time_of_laser_spot_img_right = para_dict_right["exposure_time_of_laser_spot_img_right"] 
        path_main = para_dict_right["path_main"] 

        if (self.camera_obj_right.cam_obj.IC_IsDevValid(self.camera_obj_right.hGrabber_obj)):
            self.camera_obj_right.cam_obj.IC_StartLive(self.camera_obj_right.hGrabber_obj, 1)
        exposure_time_of_ref_img_right = para_dict_right["exposure_time_of_ref_img_right"]
        exposure_time_of_laser_spot_img_right = para_dict_right["exposure_time_of_laser_spot_img_right"]
        self.camera_obj_right.set_exposure_time(exposure_time_of_ref_img_right)
        time.sleep(0.5)
        img_ref_right = self.camera_obj_right.get_one_image()
        self.camera_obj_right.set_exposure_time(input_exposure_time=exposure_time_of_laser_spot_img_right)
        img_of_low_exposure_right = self.camera_obj_right.get_one_image()
        
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(img_ref_right)
        plt.subplot(1, 2, 2)
        plt.imshow(img_of_low_exposure_right)
        plt.show()

        flag_save_base_img = input("save the reference image (yes or no)?")
        if flag_save_base_img == "yes":
            path_img_reference = path_main + "img_base_right.jpg"
            path_img_low_exposure = path_main + "img_low_exposure_right.jpg"
            cv2.imwrite(path_img_reference, img_ref_right)
            cv2.imwrite(path_img_low_exposure, img_of_low_exposure_right)

    def cam_vimba_init(self, exposure_time_base = 20000, exposure_time_laser = 500): 
        """set up the camera and capture an image"""

        # set up the vimba() task 
        # camera logging
        vimba = Vimba()
        vimba.startup() 
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        camera_ids = vimba.getCameraIds()
        for cam_id in camera_ids:
            print("Camera found: ", cam_id)
        self.c0 = vimba.getCamera(camera_ids[0])
        self.c0.openCamera()
        # set the exposure time
        self.c0.ExposureTimeAbs = exposure_time_base
        try:
            self.c0.StreamBytesPerSecond = 100000000
        except:
            pass

        # camera setting
        # exposure time 
        # color channel
        # droppedframes = []
        self.c0.PixelFormat = "BGR8Packed"  
        # Creates and returns a new frame object 
        self.frame0 = self.c0.getFrame()
        # Should be called after the frame is created.
        self.frame0.announceFrame()
        # Prepare the API for incoming frames.
        self.c0.startCapture()
        # Queue frames that may be filled during frame capturing.
        self.frame0.queueFrameCapture()
        # acquisition TODO: check this function
        self.c0.runFeatureCommand("AcquisitionStart")
        # capture a quick-realtime image with an acquisition time 
        self.c0.runFeatureCommand("AcquisitionStop")
        #  Wait for a queued frame to be filled (or dequeued).
        self.frame0.waitFrameCapture(1000)
        # formulate an image
        frame_data0 = self.frame0.getBufferByteData() 
        img0 = np.ndarray(buffer=frame_data0,
                        dtype=np.uint8,
                        shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
        # save the image 
        img_base = img0
        # cv2.imwrite(path_main + "img_base.jpg", img_base)
        # cv2.imwrite("test_frame.jpg", img_base)
        # plt.imshow(img_base)
        # plt.show()
        # exit()
        # TODO: save the image in a different folder 

        # reset the camera to a lower exposure mode
        # self.c0.ExposureTimeAbs = exposure_time_laser

        return img_base

    def cam_vimba_init_with_high_exposure(self, exposure_time_base = 20000, exposure_time_laser = 500):
        """capture the image exposure time"""

        # set up the vimba() task 
        # camera logging
        vimba = Vimba()
        vimba.startup() 
        system = vimba.getSystem()
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        camera_ids = vimba.getCameraIds()
        for cam_id in camera_ids:
            print("Camera found: ", cam_id)
        self.c0 = vimba.getCamera(camera_ids[0])
        self.c0.openCamera()
        # set the exposure time
        self.c0.ExposureTimeAbs = exposure_time_base
        try:
            self.c0.StreamBytesPerSecond = 100000000
        except:
            pass

        # camera setting
        # exposure time 
        # color channel
        # droppedframes = []
        self.c0.PixelFormat = "BGR8Packed"  
        # Creates and returns a new frame object 
        self.frame0 = self.c0.getFrame()
        # Should be called after the frame is created.
        self.frame0.announceFrame()
        # Prepare the API for incoming frames.
        self.c0.startCapture()
        # Queue frames that may be filled during frame capturing.
        self.frame0.queueFrameCapture()
        # acquisition TODO: check this function
        self.c0.runFeatureCommand("AcquisitionStart")
        # capture a quick-realtime image with an acquisition time 
        self.c0.runFeatureCommand("AcquisitionStop")
        #  Wait for a queued frame to be filled (or dequeued).
        self.frame0.waitFrameCapture(1000)
        # formulate an image
        frame_data0 = self.frame0.getBufferByteData() 
        img_tmp = np.ndarray(buffer=frame_data0,
                                        dtype=np.uint8,
                                        shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
        img_base_high_exp = img_tmp.copy()
        
        return img_base_high_exp

    def cam_vimba_agent(self, exposure_time_laser = 500): 
        """base on the image first"""

        # second stage
        time.sleep(1.0)
        # reset the camera to a lower exposure mode
        self.c0.ExposureTimeAbs = exposure_time_laser
        self.frame0.queueFrameCapture()
        self.c0.runFeatureCommand("AcquisitionStart")
        self.c0.runFeatureCommand("AcquisitionStop")
        self.frame0.waitFrameCapture(1000)
        frame_data0 = self.frame0.getBufferByteData() 
        img_tmp = np.ndarray(buffer=frame_data0,
                                      dtype=np.uint8,
                                      shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
        img_base_low_exp = np.zeros(img_tmp.shape)
        img_base_low_exp = img_tmp.copy()

        return img_base_low_exp

    def raster_scan_core(self, para_dict = []):
    
        # settings 
        mode_punch_biopsy = para_dict["mode_punch_biopsy"]
        mode_surface_scan_under_oct = para_dict["mode_surface_scan_under_oct"]

        # color camera mode 
        mode_rgbd_color_cam = para_dict["mode_rgbd_color_cam"]
        path_color_img = para_dict["path_color_img"]
        cam_rgbd_color_exposure = para_dict["cam_rgbd_color_exposure"]

        # go to the fix position mode 
        idx_stop = para_dict["idx_stop"]
        mode_fix_index_pos = para_dict["mode_fix_index_pos"]

        # laser on and off mode 
        mode_laser_on_and_off = para_dict["mode_laser_on_and_off"]

        # vimba camera model
        mode_vimba_camera = para_dict["mode_vimba_camera"]

        mode_robot = para_dict["mode_robot"]
        path_main = para_dict["path_main"]
        time_stop_each_iter = para_dict["time_stop_each_iter"]
        mode_id = para_dict["mode_id"]
        laser_current_id = para_dict["laser_current_id"]
        integration_time = para_dict["integration_time"]

        path_tumorid_data = para_dict["path_tumorid_data"]
        path_tumorid_img_left = para_dict["path_tumorid_img_left"]
        path_tumorid_img_right = para_dict["path_tumorid_img_right"]
        mode_scangrid_pattern = para_dict["mode_scangrid_pattern"]
        mode_stage = para_dict["mode_stage"]
        x_init_pos_stage = para_dict["x_init_pos_stage"]
        y_init_pos_stage = para_dict["y_init_pos_stage"]
        x_final_pos_stage = para_dict["x_final_pos_stage"]
        y_final_pos_stage = para_dict["y_final_pos_stage"]
        x_step_size = para_dict["x_step_size"]
        y_step_size = para_dict["y_step_size"]
        x_max_stage = para_dict["x_max_stage"]
        y_max_stage = para_dict["y_max_stage"]
        mode_oct = para_dict["mode_oct"]
        file_name_oct_serial = para_dict["file_name_oct_serial"]
        mode_laser_green = para_dict["mode_laser_green"]
        mode_laser_red = para_dict["mode_laser_red"]
        mode_cam_left = para_dict["mode_cam_left"]
        exposure_time_of_ref_img_left = para_dict["exposure_time_of_ref_img_left"]
        exposure_time_of_laser_spot_img_left = para_dict["exposure_time_of_laser_spot_img_left"]
        mode_cam_right = para_dict["mode_cam_right"]
        exposure_time_of_ref_img_right = para_dict["exposure_time_of_ref_img_right"]
        exposure_time_of_laser_spot_img_right = para_dict["exposure_time_of_laser_spot_img_right"]  
        path_json = path_main + 'para_exp.json'
        
        # print("path_json = ", path_json)
        # with open(path_json, 'w') as fp:
        #     json.dump(para_dict, fp)

        # load the robot trajectory
        # q_roi_init = para_dict["q_roi_init"] 
        q_home = para_dict["q_home"] 
        q_setpoint_1 = para_dict["q_setpoint_1"]
        q_setpoint_2 = para_dict["q_setpoint_2"]
        idx_align_center = para_dict["idx_align_center"]

        # load the pre-defined trajectory
        time_to_home = para_dict["time_to_home"]
        time_to_setpoint_1 = para_dict["time_to_setpoint_1"]
        time_to_first_pos = para_dict["time_to_first_pos"] 
        traj_robot_config = para_dict["traj_robot_config"] 

        # scanning grid
        para_scan_grid = {}
        para_scan_grid["x_init_pos_stage"] = x_init_pos_stage
        para_scan_grid["y_init_pos_stage"] = y_init_pos_stage
        para_scan_grid["x_final_pos_stage"] = x_final_pos_stage
        para_scan_grid["y_final_pos_stage"] = y_final_pos_stage
        para_scan_grid["x_step_size"] = x_step_size
        para_scan_grid["y_step_size"] = y_step_size
        para_scan_grid["mode_scangrid_pattern"] = mode_scangrid_pattern
        x_scan_grid, y_scan_grid = self.get_scanning_grid(para_dict=para_scan_grid)
        para_dict["x_scan_grid"] = x_scan_grid
        para_dict["y_scan_grid"] = y_scan_grid        

        """laser pointer"""
        if mode_laser_green == "true": 
            time.sleep(0.5)
            self.laser_diode.on_laser_green() 
        if mode_laser_red == "true":
            self.laser_diode.on_laser_red() 
            time.sleep(0.5)

        """oct"""
        if mode_oct == "true":
            print("oct serial initalization")
            input("move the cover and start oct scan")
            oct_token = self.oct.octvolscan(file_name_oct_serial)
            if oct_token == False:
                print("oct scan failed")
                return 0

        """vimba camera"""
        # TODO: replace the camera with a new setting
        # not with the camera -> testing the configuration
        # mode_vimba_camera = "false" 
        if mode_vimba_camera == "true":
            
            # set up the vimba() task 
            # camera logging
            vimba = Vimba()
            vimba.startup() 
            system = vimba.getSystem()
            system.runFeatureCommand("GeVDiscoveryAllOnce")
            camera_ids = vimba.getCameraIds()
            for cam_id in camera_ids:
                print("Camera found: ", cam_id)
            self.c0 = vimba.getCamera(camera_ids[0])
            self.c0.openCamera()
            # set the exposure time
            exposure_time_base = 20000
            exposure_time_laser = 500        
            self.c0.ExposureTimeAbs = exposure_time_base
            try:
                self.c0.StreamBytesPerSecond = 100000000
            except:
                pass

            # camera setting
            # exposure time 
            # color channel
            # droppedframes = []
            self.c0.PixelFormat = "BGR8Packed"  
            # Creates and returns a new frame object 
            self.frame0 = self.c0.getFrame()
            # Should be called after the frame is created.
            self.frame0.announceFrame()
            # Prepare the API for incoming frames.
            self.c0.startCapture()
            # Queue frames that may be filled during frame capturing.
            self.frame0.queueFrameCapture()
            # acquisition TODO: check this function
            self.c0.runFeatureCommand("AcquisitionStart")
            # capture a quick-realtime image with an acquisition time 
            self.c0.runFeatureCommand("AcquisitionStop")
            #  Wait for a queued frame to be filled (or dequeued).
            self.frame0.waitFrameCapture(1000)
            # formulate an image
            frame_data0 = self.frame0.getBufferByteData() 
            img0 = np.ndarray(buffer=frame_data0,
                            dtype=np.uint8,
                            shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
            # save the image 
            img_base = img0
            cv2.imwrite(path_main + "img_base.jpg", img_base)
            # cv2.imwrite("test_frame.jpg", img_base)
            plt.imshow(img_base)
            plt.show()
            # exit()
            # TODO: save the image in a different folder 

            # reset the camera to a lower exposure mode
            self.c0.ExposureTimeAbs = exposure_time_laser

        """camera"""
        # color camera 
        if mode_rgbd_color_cam == "true": 

            self.cam_rgbd_color.cam_init(mode_of_exposure="low_exposure", val_exposure = 800.00)
            time.sleep(0.5)
            img_base = self.cam_rgbd_color.color_img_single()
            cv2.imwrite(path_main + "img_base.jpg", img_base)
            # capture a good base image
            print("rgbd-color camera initialization")

            # self.cam_rgbd_color.cam_init(mode_of_exposure="low_exposure", val_exposure = cam_rgbd_color_exposure)
            self.cam_rgbd_color.exposure_adjust(val_exposure = cam_rgbd_color_exposure)
            time.sleep(0.5)

        # left camera
        if mode_cam_left == "true":
            print("left camera initialization")
            input("left image (remove oct)")
            para_dict_left = {} 
            para_dict_left["exposure_time_of_ref_img_left"] = exposure_time_of_ref_img_left
            para_dict_left["exposure_time_of_laser_spot_img_left"] = exposure_time_of_laser_spot_img_left
            para_dict_left["path_main"] = path_main 
            self.cam_left_init(para_dict_left=para_dict_left) 
        # right camera 
        if mode_cam_right == "true":
            print("right camera initialization")
            input("right image (remove oct)")
            para_dict_right = {} 
            para_dict_right["exposure_time_of_ref_img_right"] = exposure_time_of_ref_img_right
            para_dict_right["exposure_time_of_laser_spot_img_right"] = exposure_time_of_laser_spot_img_right
            para_dict_right["path_main"] = path_main 
            self.cam_right_init(para_dict_right=para_dict_right) 

        """tumorid"""
        if mode_id == "true":
            self.tumorid.clear_error()
            self.tumorid.set_integration_time(integration_time=integration_time)

        # if mode_stage == "false" and mode_id == "false":
        #     self.tumorid.turn_laser_OFF()
        #     return 0

        """robot stage movement -> OCT scan is 11.2 mm"""
        if x_init_pos_stage > x_max_stage or y_init_pos_stage > y_max_stage or x_final_pos_stage > x_max_stage or y_final_pos_stage > y_max_stage:
            print("incorrect stage inputs")
            return 0

        # robot arm configuration  
        if mode_robot == "true": 

            # TODO: initialize the robot in a global perspective
            # initialized the robot configuration
            counter = 0
            input("press enter to start the robot movement")
            self.robot.init_planning_robot() 
            self.robot.init_physical_robot(q_init=q_home)
            input("press to move down")

            # start the robot
            # update the workflow: no need to restart the program 
            time.sleep(0.5)
            # self.robot.robotControlApi.start()

            # move to the first setpoint-1
            # TODO: use the configuration minimization method to speed up the process
            # input("move to the setpoint-1")
            self.robot.constantVServo( self.robot.robotControlApi, 
                                       time_to_setpoint_1, 
                                       self.robot.klampt_2_controller( q_setpoint_1 ), 0.004)
            time.sleep(1.0)

            # move the robot to the first position
            # input("move to the centerized alignment center (with the index)")
            
            # if mode_surface_scan_under_oct == "true": 
            print("time_to_first_pos = ", time_to_first_pos)
            input("check and continue")
            idx_align_center = 0
            self.robot.constantVServo( self.robot.robotControlApi, 
                                       time_to_first_pos, 
                                       self.robot.klampt_2_controller( traj_robot_config[idx_align_center].tolist() ), 0.004)
            time.sleep(1.0)
            # input("align the center point (id = 0) with the tissue center")
            print("start to move back to the first point and start scan")

            if FLAG_TUMORID_TRAJ_TEST == "true":
                input("enter to start the laser")
                self.tumorid.turn_laser_ON(laser_current = laser_current_id)
                time.sleep(time_stop_each_iter)

            # load the robot trajectory
            list_of_integration_time = [] 
            idx_start_pos = 0
            t1_robot_global = time.time() 
            for idx_move, config_tmp in enumerate(traj_robot_config): 

                if mode_fix_index_pos == "true": 
                    if idx_move == idx_stop:
                        # this is after finishing the 2-th index. 
                        # stop the robot
                        self.robot.robotControlApi.stop()
                        input("stop at the index position of " + str(idx_move))
                        exit()

                # move to the target position
                if mode_robot == "true" and (idx_move >= idx_start_pos):
                    self.robot.constantVServo(self.robot.robotControlApi,
                                              time_stop_each_iter,
                                              self.robot.klampt_2_controller( traj_robot_config[idx_move].tolist() ), 
                                              0.004)
                    print("The current point is ", idx_move)
                else:
                    continue

                if idx_move == idx_start_pos:
                    input("press to start the scanning")
                time.sleep(time_stop_each_iter)

                # id + camera 
                if counter == 0 and mode_id == "true":
                    
                    if mode_laser_on_and_off == "false":
                        # global laser on mode 
                        input("enter to start the laser")
                        self.tumorid.turn_laser_ON(laser_current = laser_current_id)
                        time.sleep(time_stop_each_iter)
                    else: 
                        # laser on local mode (reset in a loop)
                        input("enter to start the laser")

                    if mode_punch_biopsy == "true":

                        # switch on the laser to move to the center of the tumor region 
                        input("press to laser on, move laser to the tumor center")
                        laser_current_id_init = 0.13
                        self.tumorid.turn_laser_ON(laser_current = laser_current_id_init)
                        input("press to stop the laser")
                        self.tumorid.turn_laser_OFF()

                        # # a second base image
                        # input("after alignment press to capture the second based image")
                        # self.frame0.queueFrameCapture()
                        # self.c0.runFeatureCommand("AcquisitionStart")
                        # self.c0.runFeatureCommand("AcquisitionStop")
                        # self.frame0.waitFrameCapture(1000)
                        # frame_data0 = self.frame0.getBufferByteData() 
                        # img_base_second = np.ndarray(buffer=frame_data0,
                        #                 dtype=np.uint8,
                        #                 shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                        # cv2.imwrite(path_main + "img_base_second.jpg", img_base_second)

                # diode ik mode 
                if FLAG_LASER_diode_IK == "true": 
                    # capture the images 
                    # TODO: implement the function 
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + ".jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + ".jpg", img_tmp)

                if FLAG_LASER_FIBER_CALIB == "true":
                    # before-laser -> capture the image
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + "_atlaser.jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + "_atlaser.jpg", img_tmp)
                    input("enter to move to next step")

                if FLAG_DIODE_TRAJ_TEST == "true" or FLAG_TUMORID_TRAJ_TEST == "true" or FLAG_EXVIVO_TISSUE_SCAN_WITH_DIODE == "true":

                    # input("stop to adjust the calibration board")
                    # input("stop and capture the images")
                    # before-laser -> capture the image
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + "_atlaser.jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + "_atlaser.jpg", img_tmp)
                        

                # calibration: fid test
                # diode
                if FLAG_LASER_DIODE_FID_TEST == "true":

                    input("stop to adjust the calibration board")
                    input("stop and capture the images")
                    # before-laser -> capture the image
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + "_atlaser.jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + "_atlaser.jpg", img_tmp)

                    # oct mode
                    input("enter to stop moving and start oct")
                    if mode_oct == "true" and FLAG_LASER_DIODE_FID_TEST == "true":
                        """oct with each fiducial after alignments (laser spots)"""
                        print("oct serial initalization")
                        input("move the cover and start oct scan")
                        oct_token = self.oct.octvolscan(file_name_oct_serial)
                        if oct_token == False:
                            print("oct scan failed")
                            return 0
                    
                    # check the oct file name
                    input("check please relabel the oct file name")

                # calibration mode
                # tumorid  
                if FLAG_LASER_INCIDENT_DIR_CALIB == "true" or FLAG_LASER_TUMORID_FID_TEST == "true": 
                    
                    input("press to start the low-laser")
                    laser_current_id_init = 0.120
                    self.tumorid.turn_laser_ON(laser_current = laser_current_id_init)                    
                    input("stop to adjust the calibration board")
                    input("stop and capture the images")
                    # before-laser -> capture the image
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + "_atlaser.jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + "_atlaser.jpg", img_tmp)

                    input("enter to stop the laser")
                    self.tumorid.turn_laser_OFF()
                    time.sleep(0.5)
                    input("enter to capture the images")
                    # post-laser -> capture the image
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + "_postlaser.jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + "_postlaser.jpg", img_tmp)

                    if mode_oct == "true":
                        if FLAG_LASER_TUMORID_FID_TEST == "true" or FLAG_LASER_INCIDENT_DIR_CALIB == "true": 
                            """oct with each fiducial after alignments (laser spots)"""
                            print("oct serial initalization")
                            input("move the cover and start oct scan")
                            oct_token = self.oct.octvolscan(file_name_oct_serial)
                            if oct_token == False:
                                print("oct scan failed")
                                return 0
                        
                    # check the oct file name
                    input("check please relabel the oct file name")

                if mode_oct == "true" and FLAG_DIODE_INCIDENT_DIR_CALIB == "true": 
                    
                    input("stop to adjust the calibration board")
                    input("stop and capture the images")
                    # before-laser -> capture the image
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + "_atlaser.jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + "_atlaser.jpg", img_tmp)
                    
                    """oct with each fiducial after alignments (laser spots)"""
                    print("oct serial initalization")
                    input("move the cover and start oct scan")
                    oct_token = self.oct.octvolscan(file_name_oct_serial)
                    if oct_token == False:
                        print("oct scan failed")
                        return 0

                # calibration of the axis-only -> this is im
                if mode_oct == "true" and FLAG_LASER_AXIX_ORIENTATION_CALIB == "true":
                    """oct with each fiducial after alignments (laser spots)"""
                    print("oct serial initalization")
                    input("move the cover and start oct scan")
                    oct_token = self.oct.octvolscan(file_name_oct_serial)
                    if oct_token == False:
                        print("oct scan failed")
                        return 0

                # rgbd-camera 
                if mode_rgbd_color_cam == "true":
                    img_color_tmp = self.cam_rgbd_color.color_img_single()
                    cv2.imwrite(path_color_img + str(counter) + ".jpg", img_color_tmp)
                    
                # tumorid
                if mode_id == "true":

                    # TODO: check time-in-loop or time-at-start 
                    # whether to reset the integration time all the time in the loop 
                    # note: not suggested as not necessary to reset the device, if needed, this is feasible 
                    time.sleep(0.5)     # stop for the robot to move 
                    self.tumorid.set_integration_time(integration_time=integration_time)

                    # TODO: check the thorblabs laser setting device 
                    # buffer time: 0.50 seconds to wait until the laser current is stablized. 
                    if mode_laser_on_and_off == "true":
                        self.tumorid.turn_laser_ON(laser_current=laser_current_id)
                        time.sleep(0.50)

                    # label the fiducial 
                    # input("low-laser on to label the fid")

                    # capture the image while the laser is on
                    if mode_vimba_camera == "true":
                        print("begin to capture the image") 
                        # start the camera capture 
                        self.frame0.queueFrameCapture()
                        self.c0.runFeatureCommand("AcquisitionStart")
                        self.c0.runFeatureCommand("AcquisitionStop")
                        #  Wait for a queued frame to be filled (or dequeued).
                        self.frame0.waitFrameCapture(1000)
                        # formulate an image
                        frame_data0 = self.frame0.getBufferByteData() 
                        img0 = np.ndarray(buffer=frame_data0,
                                        dtype=np.uint8,
                                        shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                        print("img0 tmp = ", img0.shape)

                        # save in a better image -> fix this code
                        cv2.imwrite(path_color_img + str(counter) + ".jpg", img0)

                    # camera image during the laser is on 
                    if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + ".jpg", img_tmp)
                    if mode_cam_right == "true": 
                        img_tmp = self.camera_obj_right.get_one_image()
                        cv2.imwrite(path_tumorid_img_right + str(counter) + ".jpg", img_tmp)
                        
                    # collect the spectrum data.
                    t1 = time.time()
                    spectrum_data = self.tumorid.get_spectrum_single_data()
                    t2 = time.time() 
                    time_integration_tmp = t2 - t1
                    print("integration time = ", time_integration_tmp)
                    list_of_integration_time.append(time_integration_tmp)
                    print("finish the tumorid capture")

                    # TODO: turn off the laser officially 
                    # buffer time to switch off the laser (not necessary) 
                    # time.sleep(0.25)
                    if mode_laser_on_and_off == "true": 
                        self.tumorid.turn_laser_OFF()
                    
                    # TODO: get the wavelength for each unit section scanning
                    # out of the loop of the spectrum collection 
                    wavelength_data = self.tumorid.get_wavelength_data()

                    # save the data 
                    np.save(path_tumorid_data + "spectrum_" + str(counter) + ".npy", spectrum_data)
                    np.save(path_tumorid_data + "wavelength_" + str(counter) + ".npy", wavelength_data)

                if mode_punch_biopsy == "true": 
                    
                    # step-1: write the point index 
                    print("after tumorid capture, write the point index")
                    id_number = idx_move + 1 # input("input the punch id_number = ")
                    # id_letter = input("input the letter number = ")
                    
                    print("idx_move = ", idx_move)
                    print("id_number = ", id_number)
                    input("check")

                    # step-2: turn the the laser 
                    if mode_laser_on_and_off == "true":

                        input("press to turn on the laser")
                        laser_current_id_label = 0.13
                        self.tumorid.turn_laser_ON(laser_current=laser_current_id_label)
                        input("keep laser on until label")
                        print("laser on")
                        
                    # step-3: start for image capture
                    input("press to capture the pre-image (low laser light on)")
                    self.frame0.queueFrameCapture()
                    self.c0.runFeatureCommand("AcquisitionStart")
                    self.c0.runFeatureCommand("AcquisitionStop")
                    self.frame0.waitFrameCapture(1000)
                    frame_data0 = self.frame0.getBufferByteData() 
                    img0 = np.ndarray(buffer=frame_data0,
                                    dtype=np.uint8,
                                    shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                    cv2.imwrite(path_color_img + "punch_" + str(idx_move) + "_prelabel" + ".jpg", img0)
                    
                    # TODO: 
                    input("label the tissue point with dye")
                    # input("remove the tool and prepare for the postlabel image")

                    if mode_laser_on_and_off == "true":
                        input("press to turn off the laser")
                        self.tumorid.turn_laser_OFF()
                        print("laser off")

                    # # get the post image capture (after this process)
                    # input("get the post image capture")
                    # self.frame0.queueFrameCapture()
                    # self.c0.runFeatureCommand("AcquisitionStart")
                    # self.c0.runFeatureCommand("AcquisitionStop")
                    # self.frame0.waitFrameCapture(1000)
                    # frame_data0 = self.frame0.getBufferByteData() 
                    # img0 = np.ndarray(buffer=frame_data0,
                    #                 dtype=np.uint8,
                    #                 shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                    # cv2.imwrite(path_color_img + "punch_" + str(id_number) + "_" + str(id_letter) + "_postlabel" + ".jpg", img0)
                    # print("finish the postlabel image")

                # update the counter
                counter = counter + 1

            # save the integrtaion time (for verification)
            # save the inetgration time mode (this is important)
            # np.save(path_main + "list_of_integration_time.npy", list_of_integration_time)

            # countg the time 
            t2_robot_global = time.time() 
            time_of_robot_running = t2_robot_global - t1_robot_global
            print("total running time = ", time_of_robot_running)
            
            # note to save the information. 
            with open(path_main + 'note.txt', 'w') as output:
                output.write("total robot running time = " + str(time_of_robot_running))

            # turn off the laser
            if mode_id == "true":
                self.tumorid.turn_laser_OFF()
                # time.sleep(time_stop_each_iter)
                time.sleep(1.0)

            if FLAG_TUMORID_TRAJ_TEST == "true":
                self.tumorid.turn_laser_OFF()
                time.sleep(1.0)

            # # return to the set-point position 
            if FLAG_LASER_AXIX_ORIENTATION_CALIB == "true": 
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)
            
            if FLAG_LASER_INCIDENT_DIR_CALIB == "true": 
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

            if FLAG_LASER_FIBER_CALIB == "true":

                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

                if mode_cam_left == "true":
                        img_tmp = self.camera_obj_left.get_one_image()
                        cv2.imwrite(path_tumorid_img_left + str(counter) + "_atlaser.jpg", img_tmp)
                if mode_cam_right == "true": 
                    img_tmp = self.camera_obj_right.get_one_image()
                    cv2.imwrite(path_tumorid_img_right + str(counter) + "_atlaser.jpg", img_tmp)
                input("enter to move to next step")

            if FLAG_LASER_FIBER_SPEED_TEST == "true":
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

            if FLAG_LASER_FIBER_MICE_EXP == "true":
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

            if FLAG_DIODE_TRAJ_TEST == "true" or FLAG_TUMORID_TRAJ_TEST == "true":
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

            if FLAG_EXVIVO_TISSUE_SCAN == "true":
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

            if FLAG_EXVIVO_TISSUE_SCAN_WITH_DIODE == "true":
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

            if FLAG_SURFACE_SCAN == "true":
                input("move to setpoint-2")
                self.robot.move_to(self.robot.robotControlApi, time_to_setpoint_1, self.robot.klampt_2_controller( q_setpoint_2 ), 0.004)

            # stop th laser 
            input("setpoint stop the laser and move back")

            # return to the home position
            # input("move to the home position")
            self.robot.move_to(self.robot.robotControlApi, time_to_home, self.robot.klampt_2_controller( q_home ), 0.004)

             # take the post image with all the color dyes
            if mode_punch_biopsy == "true":
                exposure_time_laser = 20000      # exposure time adjusted to 1500
                self.c0.ExposureTimeAbs = exposure_time_laser
                # capture the frame 
                self.frame0.queueFrameCapture()
                # acquisition TODO: check this function
                self.c0.runFeatureCommand("AcquisitionStart")
                # capture a quick-realtime image with an acquisition time 
                self.c0.runFeatureCommand("AcquisitionStop")
                #  Wait for a queued frame to be filled (or dequeued).
                self.frame0.waitFrameCapture(1000)
                # formulate an image
                frame_data0 = self.frame0.getBufferByteData() 
                img0 = np.ndarray(buffer=frame_data0,
                                dtype=np.uint8,
                                shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                # save the image 
                img_base = img0
                cv2.imwrite(path_main + "img_last.jpg", img_base)

            # stop the robot
            time.sleep(0.5)
            self.robot.robotControlApi.stop()
            print("robot being stopped")

            # TODO: stop the camera 
            self.c0.endCapture()
            self.c0.revokeAllFrames()
            self.c0.closeCamera()

            return 0 

        # test the stage with random positions
        if mode_stage == "true": 
        
            input("press enter to start the stage movement")
            list_of_x = []
            list_of_y = []
            counter = 0
            assert len(x_scan_grid) == len(y_scan_grid)
            for idx_pos_tmp in range(len(x_scan_grid)):

                # get the current position (from the mesh-grid data)
                x_pos_tmp = x_scan_grid[idx_pos_tmp]
                y_pos_tmp = y_scan_grid[idx_pos_tmp]
                list_of_x.append(x_pos_tmp)
                list_of_y.append(y_pos_tmp)

                # move to the target position
                if mode_stage == "true":
                    self.stage.move_to(x_pos_tmp, y_pos_tmp)
                    print("at x-pos, y-pos = ", x_pos_tmp, y_pos_tmp)
                if idx_pos_tmp == 0:
                    input("press to start the scanning")
                time.sleep(time_stop_each_iter)

                # id + camera 
                if counter == 0 and mode_id == "true":
                    input("enter to start the laser")
                    self.tumorid.turn_laser_ON(laser_current = laser_current_id)
                    time.sleep(time_stop_each_iter)

                # camera image
                if mode_cam_left == "true":
                    img_tmp = self.camera_obj_left.get_one_image()
                    cv2.imwrite(path_tumorid_img_left + "xy_" + str(x_pos_tmp) + "_" + str(y_pos_tmp) + ".jpg", img_tmp)
                if mode_cam_right == "true": 
                    img_tmp = self.camera_obj_right.get_one_image()
                    cv2.imwrite(path_tumorid_img_right + "xy_" + str(x_pos_tmp) + "_" + str(y_pos_tmp) + ".jpg", img_tmp)

                # tumorid
                if mode_id == "true":
                    spectrum_data = self.tumorid.get_spectrum_single_data()
                    wavelength_data = self.tumorid.get_wavelength_data()
                    np.save(path_tumorid_data + "spectrum_" + str(counter) + ".npy", spectrum_data)
                    np.save(path_tumorid_data + "wavelength_" + str(counter) + ".npy", wavelength_data)

                # update the counter
                counter = counter + 1

        # save the coordinate coordinate
        # np.save(path_main + "x_robot_coordinate.npy", np.asarray(list_of_x))
        # np.save(path_main + "y_robot_coordinate.npy", np.asarray(list_of_y))
        
        # stop the laser
        if mode_id == "true":
            self.tumorid.turn_laser_OFF()

        # turn off the laser
        if mode_laser_green == "true": 
            self.laser_diode.off_laser_green() 
        if mode_laser_red == "true":
            self.laser_diode.off_laser_red() 

        # stop the camera
        if mode_cam_left == "true":
            self.camera_obj_left.cam_obj.IC_StopLive(self.camera_obj_left.hGrabber_obj)
            self.camera_obj_left.cam_obj.IC_ReleaseGrabber(self.camera_obj_left.hGrabber_obj)
        if mode_cam_right == "true":
            self.camera_obj_right.cam_obj.IC_StopLive(self.camera_obj_right.hGrabber_obj)
            self.camera_obj_right.cam_obj.IC_ReleaseGrabber(self.camera_obj_right.hGrabber_obj)

        # move the stage to the origin
        if mode_stage == "true":
            input("press enter to move to home")
            x_pos_origin = 0
            y_pos_origin = 0
            self.stage.move_to(x_pos_origin, y_pos_origin)

if __name__ == "__main__":

    # class obj
    class_test = exp_unit_test()
    class_test.raster_scan_core()
