
import klampt
from klampt import * 
import numpy as np
import matplotlib.pyplot as plt 
import time
import math
import scipy.io

class robot_arm_utility_base():

    def __init__(self):

        # ip for the robot
        # self.host = [] 

        # class object (from the models)
        # self.ur5_robot = robot_arm_hardware.robot_ur5()

        # robot system configuration
        self.config_1                       = [0.0, -4.06295197, -1.0686651, 1.5598008, -0.488692, -0.10472, -3.14159, 0.0]
        self.config_2                       = [0.0, -3.89208, -1.01229, 1.62316, -0.488692, -0.10472, -3.14159, 0.0] 
        self.config_tumorid_klampt          = [0.0, -4.06295197, -1.0686651, 1.5598008, -0.488692, -0.10472, -3.14159, 0.0]
        self.config_tumorid_home_klampt     = [0.0, -4.06295197, -1.0686651, 1.5598008, -0.488692, -0.10472, -3.14159, 0.0]
        self.config_tumorid_test_sim_klampt = [0.0, -3.1555553, -2.2385593, -1.2351695, 0.32410764, 0.0445059, -5.36095333, 0.0]
        self.config_home_klampt             = [0.0, -math.pi / 2.0, 0.0, -math.pi / 2.0, 0.0, 0.0, 0.0]
        self.world                          = klampt.WorldModel()
        self.worldfilepath                  = "./database/data_test/robot_system_data/worlds/robots/ur5e.rob"
        self.xmlfile                        = './database/data_test/robot_system_data/worlds/MEDX_world.xml'

    def pts_3d_scanning_grid(self, para_dict = {}):
        """get a 2d meshgrid with unique configurations
            1. show a 2d map 
            2. show a 3d map 
        """

        x_len = para_dict["x_len"]
        y_len = para_dict["y_len"]
        x_center = para_dict["x_center"]
        y_center = para_dict["y_center"]
        num_of_pts_per_line = para_dict["num_of_pts_per_line"]
        z_height_scanning_grid = para_dict["z_height_scanning_grid"]
        flag_vis = para_dict["flag_vis"]

        # step-1: scanning grid in a 2d mesh
        x_len_half = x_len / 2.0
        y_len_half = y_len / 2.0
        x1 = x_center - x_len_half
        x2 = x_center + x_len_half
        y1 = y_center - y_len_half
        y2 = y_center + y_len_half
        x_list = np.linspace(x1, x2, num_of_pts_per_line)
        y_list = np.linspace(y1, y2, num_of_pts_per_line)

        # step-2: scanning grid in a 2d mesh
        [x_mesh, y_mesh] = np.meshgrid(x_list, y_list)
        x_mesh_use = x_mesh.ravel(order='C')
        y_mesh_use = y_mesh.ravel(order='C')

        print("x_mesh_use = ", x_mesh_use)
        print("y_mesh_use = ", y_mesh_use)
        
        num_of_row = len(x_mesh)
        if (num_of_row % 2) != 0:
            print("Odd")
        if (num_of_row % 2) == 0:
            print("Even")
        x_mesh_use = []
        y_mesh_use = []
        for i in range(num_of_row):
            if i % 2 != 0:
                x_mesh_list = x_mesh[i, :]
                y_mesh_list = y_mesh[i, :]
            if i % 2 == 0:
                # flip
                x_mesh_list = np.flip(x_mesh[i, :])
                y_mesh_list = np.flip(y_mesh[i, :])
            x_mesh_use = np.concatenate((x_mesh_use, x_mesh_list), axis=0)
            y_mesh_use = np.concatenate((y_mesh_use, y_mesh_list), axis=0)
        z_mesh_use = np.ones((len(x_mesh_use[:]), 1)) * z_height_scanning_grid
        z_mesh_use = z_mesh_use.ravel()
        pts_grid = np.zeros((len(z_mesh_use[:]), 3))
        for i in range(len(z_mesh_use[:])):
            print("x_mesh_use = ", x_mesh_use[i])
            print("y_mesh_use = ", y_mesh_use[i])
            print("z_mesh_use = ", z_mesh_use[i])
            print(np.asarray([x_mesh_use[i], y_mesh_use[i], z_mesh_use[i]]))
            pts_grid[i, :] = np.asarray([x_mesh_use[i], y_mesh_use[i], z_mesh_use[i]])

        # vis 
        if flag_vis == "true":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pts_grid[:,0], pts_grid[:,1], pts_grid[:,2], c = 'b', marker='o')
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_zlabel('Z-axis')
            plt.show()

        return pts_grid

    def traj_scan_exp(self, para_dict): 
        """surface trajectory model
        1. scan the surface trajectory 
        2. move the models"""

        # camera exposure time 
        # adjust the exposure time
        # self.obj_rgbd.Caminit(mode="low_exposure", val_exposure = 300.0)
        # time.sleep(0.5)

        # config_traj_robot, flag_save = "true"
        traj_robot_config = para_dict["traj_robot_config"]
        flag_save = para_dict["flag_save"]
        t_stepsize = para_dict["t_stepsize"]
        idx_start_pos = para_dict["idx_start_pos"]
        time_to_first_pos = para_dict["time_to_first_pos"]
        time_of_loop = para_dict["time_of_loop"]
        q_init = para_dict["q_init"]
        q_home = para_dict["q_home"]

        print("config_traj_robot = ", traj_robot_config)
        print("t_stepsize = ", t_stepsize)
        print("time_to_first_pos = ", time_to_first_pos)
        print("time_of_loop = ", time_of_loop)
        input("press to move to the next point")

        # Initialization of the simulation with the current configuration
        self.ur5_robot.init_planning_robot()

        # initilaize the physical robot for the initial planning
        # q_init = [0.0, -4.06295197, -1.0686651, 1.5598008, -0.488692, -0.10472, -3.14159, 0.0]
        self.ur5_robot.init_physical_robot(q_init=q_init)

        # step-2: move to the first scanning position
        count = 0
        input("move to start the robot position")
        self.ur5_robot.robotControlApi.start()
        config_qs = traj_robot_config
        self.ur5_robot.constantVServo( self.ur5_robot.robotControlApi, time_to_first_pos, self.ur5_robot.klampt_2_controller( traj_robot_config[idx_start_pos] ), 0.004)
        time.sleep(t_stepsize)

        # collect the first image data
        # if flag_save == "true":
        #     self.obj_rgbd.ID = count
        #     self.obj_rgbd.get_SingleData(flag_save = 1)
        #     count += 1

        # for i, item in enumerate( config_qs ):
        #     print("item = ", item)
        # # input("check to move next")

        # start from 2-th point
        for i, item in enumerate( config_qs ):
            # only move to following positions except for the first one
            if i > idx_start_pos:
                self.ur5_robot.constantVServo(self.ur5_robot.robotControlApi, time_of_loop, self.ur5_robot.klampt_2_controller( item ), 0.004)
                time.sleep(t_stepsize)
                # TODO: update the sensor data herein. 
                time.sleep(t_stepsize)
                count += 1

        # move to the original position 
        time_to_home = 3
        self.ur5_robot.constantVServo(self.ur5_robot.robotControlApi, 
                                      time_to_home, 
                                      self.ur5_robot.klampt_2_controller( q_home ), 
                                      0.004)
        self.ur5_robot.robotControlApi.stop()

class robot_simulation(robot_arm_utility_base): 

    def __init__(self):
        super().__init__()

    def simulation_scanning_plane(self, para_dict = []):
        """simulation of the tumorid robot movement
        1. design a planar surface grid.
        2. simulation the robot to move to these points 
        """

        # parameters
        if len(para_dict) == 0:
            print("input the parameter (exit)")
            return 0
        config_tumorid_klampt = para_dict["config_tumorid_klampt"]
        config_tip_ee_orientation = para_dict["config_tip_ee_orientation_degree"]
        config_tip_ee_translation = para_dict["config_tip_ee_translation"]
        x_len = para_dict["x_len"] 
        y_len = para_dict["y_len"] 
        x_center = para_dict["x_center"]
        y_center = para_dict["y_center"] 
        num_of_pts_per_line = para_dict["num_of_pts_per_line"] 
        z_height_scanning_grid = para_dict["z_height_scanning_grid"] 
        path_config_save = para_dict["path_config_save"]
        flag_vis = para_dict["flag_vis"]
        flag_save = para_dict["flag_save"]

        # world robot
        path_robot_file = self.xmlfile
        res = self.world.readFile(path_robot_file)
        if not res:
            raise RuntimeError("unable to load model")
        self.robot = self.world.robot(0)
        ee_link = 7
        base_link = 0
        self.EELink = self.robot.link(ee_link)
        self.BaseLink = self.robot.link(base_link)
        self.robot.setConfig(config_tumorid_klampt)
        T_world = klampt.math.se3.identity()
        T_Base = self.BaseLink.getTransform()
        T_EE = self.EELink.getTransform()
        T_EE_init = T_EE

        # design the end-effector of the tip  
        angle_x_axis = config_tip_ee_orientation[0] / 180.0 * math.pi
        angle_y_axis = config_tip_ee_orientation[1] / 180.0 * math.pi
        angle_z_axis = config_tip_ee_orientation[2] / 180.0 * math.pi
        rotation_tip_to_ee = klampt.math.so3.from_rpy([ angle_x_axis, angle_y_axis, angle_z_axis ]) 
        T_EE2Tip = (rotation_tip_to_ee, [0.0, 0.0, 0.0])
        T_tip_translation = [ config_tip_ee_translation[0], config_tip_ee_translation[1], config_tip_ee_translation[2] ]
        
        # T_EE2Tip_zero = (klampt.math.so3.from_rpy([ 0.0, 0.0, 0.0 ]) , [0.0, 0.0, 0.0])
        T_EE2Tip_zero = ( rotation_tip_to_ee , [0.0, 0.0, 0.0])

        # scanning grid
        # TODO: simplify the model -> finished.
        para_dict_3d_grid = {}
        para_dict_3d_grid["x_len"] = x_len
        para_dict_3d_grid["y_len"] = y_len 
        para_dict_3d_grid["x_center"] = x_center
        para_dict_3d_grid["y_center"] = y_center
        para_dict_3d_grid["num_of_pts_per_line"] = num_of_pts_per_line
        para_dict_3d_grid["z_height_scanning_grid"] = z_height_scanning_grid
        para_dict_3d_grid["flag_vis"] = flag_vis
        pts_grid = self.pts_3d_scanning_grid(para_dict=para_dict_3d_grid)

        # print("x max - x min = ", np.max( pts_grid[:,0] ) - np.min( pts_grid[:,0] ) )
        # print("y max - y min = ", np.max( pts_grid[:,1] ) - np.min( pts_grid[:,1] ) )
        # exit()

        # saving mode
        # flag_save_grid_data = input("save the grid data? (yes or no)")
        # if flag_save_grid_data == "yes":
        #     #np.save("./data_calib/pts_grid_buffer_data.npy", pts_grid)
        #     print("pts_grid = ", pts_grid)
        #     input("check")

        # simulation
        # the first configuration is the home-configuration (skip) 
        # getWorldPosition of the {EE} end-effector position  
        # orientation with the end-effector
        # initialTransform: 3 x 1 position of the tip position
        q0 = config_tumorid_klampt
        Qs = [q0]
        pts_global_grid = []
        pts_tip_world = self.EELink.getWorldPosition(T_tip_translation)
        pts_use_sim = pts_grid

        # TODO: get the orientation of the {EE} 
        # vectors within the local frame 
        tform_ee = self.EELink.getTransform() 
        R_ee_in_world, t_ee_in_world = tform_ee
        vec_ee_in_world_x = R_ee_in_world[0:3]
        vec_ee_in_world_y = R_ee_in_world[3:6]
        vec_ee_in_world_z = R_ee_in_world[6:9]
        offset_global_1 = ( np.asarray([0.0, 0.0, 0.0]) * 0.0 ).tolist()
        offset_global_2 = ( np.asarray(vec_ee_in_world_x) * 0.05 ).tolist()
        offset_global_3 = ( np.asarray(vec_ee_in_world_y) * 0.05 ).tolist()

        # loop for simulation
        count_success = 0 
        scale_1 = 0.0
        scale_2 = scale_3 = 0.05
        for i in range(len(pts_use_sim)):

            # current point
            pos_in_local_grid = pts_use_sim[i]  

            # The local point definition
            # this is baesd on the reference link: self.EELink
            pos1 = [0.0, 0.0, 0.0]
            pos2 = [0.05, 0.0, 0.0]
            pos3 = [0.0, 0.05, 0.0]

            # TODO: fix the orientation
            local1 = klampt.math.se3.apply(T_EE2Tip_zero, pos1)
            local2 = klampt.math.se3.apply(T_EE2Tip_zero, pos2)
            local3 = klampt.math.se3.apply(T_EE2Tip_zero, pos3)

            # The global point definition
            # Define in {W}
            # pts_tip_world: point in the {EE} in {world}
            # pts_use_sim: point in the grid 
            global1 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_1 ))  
            global2 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_2 ))
            global3 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_3 ))
            pts_global_grid.append(global1)
            goal = klampt.model.ik.objective(self.EELink, local=[local1, local2, local3], world=[global1, global2, global3])

            # if ik.solve_nearby(goal, maxDeviation = 0.2, tol = 0.001, activeDofs = self.activeDoFs):
            if klampt.model.ik.solve(goal, tol=0.00001):
                print("IK solver success")
                print(local1, local2, local3, global1, global2, global3)
                goalConfig = self.robot.getConfig()
                self.robot.setConfig(self.robot.getConfig())
                # print(pos_target)
                # print(self.robot.getConfig())
                # raw_input()
                Qs.append(self.robot.getConfig())
                count_success += 1
            else:
                print("IK solver failure")
                print(pts_grid[i])
                goalConfig = self.robot.getConfig()
                Qs.append(self.robot.getConfig())

        if count_success == (len(Qs) - 1):
            print("Configuration success")
        else:
            print("IK failed")
            exit()
        
        # save the robot configuration 
        if flag_save == "true": 
            Qs.pop(0)
            #np.save(path_config_save, Qs)
            print("saved the robot configuration trajectory")

        # Show the frame
        # Define the Trajectory
        if flag_vis == "true": 

            traj0 = klampt.model.trajectory.RobotTrajectory(self.robot, times=list(range(len(Qs))), milestones=Qs)
            
            vis.add("world", self.world)
            vis.show()
            
            # Add the points
            for idx, item in enumerate(pts_global_grid):
                vis.add(str(idx), item)
            
            while vis.shown():
                
                vis.lock()
                t = vis.animationTime()

                # Update the frames
                T_Base = self.BaseLink.getTransform()
                T_EE = self.EELink.getTransform()
                T_EE2Tip = klampt.math.se3.identity()
                (R, T) = T_EE2Tip
                # T_tip_translation = [0.0, 0.0, -0.1]
                T_EE2Tip = (rotation_tip_to_ee, T_tip_translation)
                T_Tip = klampt.math.se3.mul(T_EE, T_EE2Tip)

                # Show the frame
                vis.add("World Frame", T_world)
                vis.add("Base Frame", T_Base)
                vis.add("EE", T_EE)
                # vis.add("Tip", T_Tip)
                # vis.add("EE_init", T_EE_init)

                q = traj0.eval(t, endBehavior='loop')
                self.robot.setConfig(q)
                vis.unlock()
                time.sleep(0.01)
                print(int(t))

            # vis.spin(float('inf'))
            vis.show(False)
            vis.kill()

        return Qs

    def line_from_vector_traj(self, para_dict = []):
        """line scanning from a center to the tumor-roi""" 

        # parameters
        if len(para_dict) == 0:
            print("input the parameter (exit)")
            return 0
        
        config_tumorid_klampt       = para_dict["config_tumorid_klampt"]
        config_tip_ee_orientation   = para_dict["config_tip_ee_orientation_degree"]
        config_tip_ee_translation   = para_dict["config_tip_ee_translation"]
        # x_len = para_dict["x_len"] 
        # y_len = para_dict["y_len"] 
        x_center                    = para_dict["x_center"]
        y_center                    = para_dict["y_center"] 
        num_of_pts_per_line         = para_dict["num_of_pts_per_line"] 
        z_height_scanning_grid      = para_dict["z_height_scanning_grid"] 
        # path_config_save            = para_dict["path_config_save"]
        flag_vis                    = para_dict["flag_vis"]

        # trajectory specific-based definition
        flag_save                   = para_dict["flag_save"]
        vec_orientation             = para_dict["vec_orientation"]
        num_of_waypts               = para_dict["num_of_waypts"]
        step_size_in_line           = para_dict["step_size_in_line"]

        # a single trajectory from a center 
        # basic definition
        # define an orientation vector
        pts_scanning_traj_local     = []
        pts_grid                    = np.zeros((num_of_waypts, 3))
        for id_step in range(num_of_waypts): 
            pts_local_tmp           = id_step * step_size_in_line * vec_orientation
            pts_local_tmp[0]        = pts_local_tmp[0] + x_center
            pts_local_tmp[1]        = pts_local_tmp[1] + y_center
            pts_local_tmp[2]        = pts_local_tmp[2] + z_height_scanning_grid 
            pts_grid[id_step,:]     = np.asarray([pts_local_tmp[0], pts_local_tmp[1], pts_local_tmp[2]])
            pts_scanning_traj_local.append(pts_local_tmp)

        # vis the grids  
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts_grid[:,0], pts_grid[:,1], pts_grid[:,2], c = 'b', marker='o')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

        # world robot
        path_robot_file = self.xmlfile
        res = self.world.readFile(path_robot_file)
        if not res:
            raise RuntimeError("unable to load model")
        self.robot = self.world.robot(0)
        ee_link = 7
        base_link = 0
        self.EELink = self.robot.link(ee_link)
        self.BaseLink = self.robot.link(base_link)
        self.robot.setConfig(config_tumorid_klampt)
        T_world = klampt.math.se3.identity()
        T_Base = self.BaseLink.getTransform()
        T_EE = self.EELink.getTransform()
        T_EE_init = T_EE

        # design the end-effector of the tip  
        angle_x_axis = config_tip_ee_orientation[0] / 180.0 * math.pi
        angle_y_axis = config_tip_ee_orientation[1] / 180.0 * math.pi
        angle_z_axis = config_tip_ee_orientation[2] / 180.0 * math.pi
        rotation_tip_to_ee = klampt.math.so3.from_rpy([ angle_x_axis, angle_y_axis, angle_z_axis ]) 
        T_EE2Tip = (rotation_tip_to_ee, [0.0, 0.0, 0.0])
        T_tip_translation = [ config_tip_ee_translation[0], config_tip_ee_translation[1], config_tip_ee_translation[2] ]
        
        # T_EE2Tip_zero = (klampt.math.so3.from_rpy([ 0.0, 0.0, 0.0 ]) , [0.0, 0.0, 0.0])
        T_EE2Tip_zero = ( rotation_tip_to_ee , [0.0, 0.0, 0.0])

        # simulation
        # the first configuration is the home-configuration (skip) 
        # getWorldPosition of the {EE} end-effector position  
        # orientation with the end-effector
        # initialTransform: 3 x 1 position of the tip position
        q0 = config_tumorid_klampt
        Qs = [q0]
        pts_global_grid = []
        pts_tip_world = self.EELink.getWorldPosition(T_tip_translation)
        pts_use_sim = pts_grid  

        # TODO: get the orientation of the {EE} 
        # vectors within the local frame 
        tform_ee = self.EELink.getTransform() 
        R_ee_in_world, t_ee_in_world = tform_ee
        vec_ee_in_world_x = R_ee_in_world[0:3]
        vec_ee_in_world_y = R_ee_in_world[3:6]
        vec_ee_in_world_z = R_ee_in_world[6:9]
        offset_global_1 = ( np.asarray([0.0, 0.0, 0.0]) * 0.0 ).tolist()
        offset_global_2 = ( np.asarray(vec_ee_in_world_x) * 0.05 ).tolist()
        offset_global_3 = ( np.asarray(vec_ee_in_world_y) * 0.05 ).tolist()

        # loop for simulation
        count_success = 0 
        scale_1 = 0.0
        scale_2 = scale_3 = 0.05
        for i in range(len(pts_use_sim)):

            # current point
            pos_in_local_grid = pts_use_sim[i]  

            # The local point definition
            # this is baesd on the reference link: self.EELink
            pos1 = [0.0, 0.0, 0.0]
            pos2 = [0.05, 0.0, 0.0]
            pos3 = [0.0, 0.05, 0.0]

            # TODO: fix the orientation
            local1 = klampt.math.se3.apply(T_EE2Tip_zero, pos1)
            local2 = klampt.math.se3.apply(T_EE2Tip_zero, pos2)
            local3 = klampt.math.se3.apply(T_EE2Tip_zero, pos3)

            # The global point definition
            # Define in {W}
            # pts_tip_world: point in the {EE} in {world}
            # pts_use_sim: point in the grid 
            global1 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_1 ))  
            global2 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_2 ))
            global3 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_3 ))
            pts_global_grid.append(global1)
            goal = klampt.model.ik.objective(self.EELink, local=[local1, local2, local3], world=[global1, global2, global3])

            # if ik.solve_nearby(goal, maxDeviation = 0.2, tol = 0.001, activeDofs = self.activeDoFs):
            if klampt.model.ik.solve(goal, tol=0.00001):
                print("IK solver success")
                print(local1, local2, local3, global1, global2, global3)
                goalConfig = self.robot.getConfig()
                self.robot.setConfig(self.robot.getConfig())
                Qs.append(self.robot.getConfig())
                count_success += 1
            else:
                print("IK solver failure")
                print(pts_grid[i])
                goalConfig = self.robot.getConfig()
                Qs.append(self.robot.getConfig())

        if count_success == (len(Qs) - 1):
            print("Configuration success")
        else:
            print("IK failed")
            exit()
        
        # save the robot configuration 
        if flag_save == "true": 
            Qs.pop(0)
            #np.save(path_config_save, Qs)
            print("saved the robot configuration trajectory")

        # Show the frame
        # Define the Trajectory
        if flag_vis == "true": 
            traj0 = klampt.model.trajectory.RobotTrajectory(self.robot, times=list(range(len(Qs))), milestones=Qs)
            vis.add("world", self.world)
            vis.show()
            # Add the points
            for idx, item in enumerate(pts_global_grid):
                vis.add(str(idx), item)
            while vis.shown():
                vis.lock()
                t = vis.animationTime()

                # Update the frames
                T_Base = self.BaseLink.getTransform()
                T_EE = self.EELink.getTransform()
                T_EE2Tip = klampt.math.se3.identity()
                (R, T) = T_EE2Tip
                # T_tip_translation = [0.0, 0.0, -0.1]
                T_EE2Tip = (rotation_tip_to_ee, T_tip_translation)
                T_Tip = klampt.math.se3.mul(T_EE, T_EE2Tip)

                # Show the frame
                vis.add("World Frame", T_world)
                vis.add("Base Frame", T_Base)
                vis.add("EE", T_EE)
                # vis.add("Tip", T_Tip)
                # vis.add("EE_init", T_EE_init)

                q = traj0.eval(t, endBehavior='loop')
                self.robot.setConfig(q)
                vis.unlock()
                time.sleep(0.01)
                print(int(t))

            # vis.spin(float('inf'))
            vis.show(False)
            vis.kill()

        return Qs

class robot_unit_test():

    def __init__(self):
        self.robot_utility      = robot_arm_utility_base()
        self.robot_simulation   = robot_simulation()

    def planar_raster_scan_histopathology(self):

        """planar raster scan"""

        # parameter settings
        para_dict = {}
        para_dict["x_len"] = 0.008
        para_dict["y_len"] = 0.008
        para_dict["x_center"] = 0.0                             
        para_dict["y_center"] = 0.0                             
        para_dict["num_of_pts_per_line"] = 8                    
        para_dict["z_height_scanning_grid"] = 0.035             
        
        # save path for the trajectory
        para_dict["flag_save"] = "true"
        para_dict["flag_vis"] = "true"
        para_dict["path_config_save"] = "./database/data_mice_official/planar_raster_scan/planar_raster_scan_histopathology_degree.npy" 
        para_dict["id_traj"] = "planar_raster_scan"
        para_dict["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 

        para_dict["config_tip_ee_orientation_degree"] = [0.0, 0.0, 0.0]
        
        # robot unique configuration design
        para_dict["config_tumorid_klampt"] = [0.0, -4.875, -0.794, 1.618, -0.532, 0.101, -4.172, 0.0]
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict) 

    def planar_raster_scan_resection(self):
        """planar raster scan"""

        # parameter settings
        para_dict = {}
        para_dict["x_len"] = 0.008
        para_dict["y_len"] = 0.008
        para_dict["x_center"] = +0.047                          
        para_dict["y_center"] = -0.012                          
        para_dict["num_of_pts_per_line"] = 8                    
        para_dict["z_height_scanning_grid"] = 0.020            
        
        # save path for the trajectory
        para_dict["flag_save"] = "true"
        para_dict["flag_vis"] = "true"
        para_dict["path_config_save"] = "./database/data_mice_official/planar_raster_scan/planar_raster_scan_resection_degree.npy" 
        para_dict["id_traj"] = "planar_raster_scan"
        para_dict["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 

        para_dict["config_tip_ee_orientation_degree"] = [+40.0, 0.0, 0.0]
        
        # robot unique configuration design
        para_dict["config_tumorid_klampt"] = [0.0, -4.875, -0.794, 1.618, -0.532, 0.101, -4.172, 0.0]
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict) 

    def planar_line_scan_resection(self, path_ref = []):
        """scan a line from the center of the tissue
        1. select a center from the given robot configuration 
        2. sample different orientation vectors around the center point. 
        3. sample the trajectory for different waypoints.
        4. create different robot trajectories """
        
        para_dict = {}
        para_dict["path_config_save"] = path_ref

        # run the simulation 
        para_dict["x_len"] = 0.05 
        para_dict["y_len"] = 0.05 
        para_dict["x_center"] = +0.050          
        para_dict["y_center"] = -0.015          
        para_dict["z_height_scanning_grid"] = 0.020     
        para_dict["config_tumorid_klampt"] = [0.0, -4.875, -0.794, 1.618, -0.532, 0.101, -4.172, 0.0]
        para_dict["num_of_pts_per_line"] = 0
                                             
        # orientation vector from a givne rotation angle
        vec_orientation = np.asarray([-1.0, 0.0, 0.0])
        theta_orientation_x = 0.0 / 180.0 * math.pi
        theta_orientation_y = 0.0 / 180.0 * math.pi
        theta_orientation_z = 0.0 / 180.0 * math.pi
        rollpitchyaw = (theta_orientation_x, theta_orientation_y, theta_orientation_z)
        matrix_rotation = klampt.math.so3.from_rpy(rollpitchyaw)
        vec_orientation_use = klampt.math.so3.apply(matrix_rotation, vec_orientation)
        vec_orientation_use = vec_orientation_use / np.linalg.norm(vec_orientation_use)    
        
        para_dict["flag_save"] = "true"
        para_dict["flag_vis"] = "true"
        para_dict["config_tip_ee_orientation_degree"] = [40, 0.0, 0.0]    
        para_dict["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 
        para_dict["vec_orientation"] = vec_orientation_use
        para_dict["num_of_waypts"] = 5
        para_dict["step_size_in_line"] = 0.0015       
        self.robot_simulation.line_from_vector_traj(para_dict=para_dict)

    def planar_line_scan_histopathology(self, path_ref = []):
        """scan a line from the center of the tissue
        1. select a center from the given robot configuration 
        2. sample different orientation vectors around the center point. 
        3. sample the trajectory for different waypoints.
        4. create different robot trajectories """

        # parameter settings
        # para_dict["path_config_save"] = path_ref
        para_dict = {}
    
        # run the simulation 
        para_dict["x_len"] = 0.05 
        para_dict["y_len"] = 0.05 
        para_dict["x_center"] = 0.0          
        para_dict["y_center"] = 0.0          
        para_dict["z_height_scanning_grid"] = 0.035     
        para_dict["config_tumorid_klampt"] = [0.0, -4.875, -0.794, 1.618, -0.532, 0.101, -4.172, 0.0]
        para_dict["num_of_pts_per_line"] = 0
                                             
        # orientation vector from a givne rotation angle
        # around the 360 orientation angles 
        vec_orientation = np.asarray([-1.0, 0.0, 0.0])
        theta_orientation_x = 0.0 / 180.0 * math.pi
        theta_orientation_y = 0.0 / 180.0 * math.pi
        theta_orientation_z = 0.0 / 180.0 * math.pi
        rollpitchyaw = (theta_orientation_x, theta_orientation_y, theta_orientation_z)
        matrix_rotation = klampt.math.so3.from_rpy(rollpitchyaw)
        vec_orientation_use = klampt.math.so3.apply(matrix_rotation, vec_orientation)
        vec_orientation_use = vec_orientation_use / np.linalg.norm(vec_orientation_use)    
        # print("vec_orientation_use = ", vec_orientation_use)
        
        # line scnaning model 
        # create a trajectory from a single line model
        para_dict["flag_save"] = "true"
        para_dict["flag_vis"] = "true"
        para_dict["config_tip_ee_orientation_degree"] = [0.0, 0.0, 0.0]     
        para_dict["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 
        para_dict["vec_orientation"] = vec_orientation_use
        para_dict["num_of_waypts"] = 10
        
        # label scan 
        para_dict["step_size_in_line"] = 1.5 / 1000

        # high scan 
        # para_dict["step_size_in_line"] = 0.75 / 1000 # unit: mm 

        self.robot_simulation.line_from_vector_traj(para_dict=para_dict)

    def calib_fiber_laser_incident_dir(self):
        """fiber couple laser"""

        # get the robot configuration
        para_dict_sim = {}
        para_dict_sim["config_tip_ee_orientation_degree"] = [0.0, 0.0, 0.0]
        para_dict_sim["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 
        para_dict_sim["x_len"] = 0.005
        para_dict_sim["y_len"] = 0.005
        para_dict_sim["x_center"] = +0.008       # move to negative
        para_dict_sim["y_center"] = +0.002       # move to positive
        para_dict_sim["num_of_pts_per_line"] = 2
        para_dict_sim["z_height_scanning_grid"] = 0.0
        
        para_dict_sim["flag_save"] = "true"
        para_dict_sim["flag_vis"] = "false"
        para_dict_sim["path_config_save"] = "./data_calib/laser_fiber_incident_dir_raster.npy"

        # laser configuration
        para_dict_sim["config_tumorid_klampt"] = [0.0, -4.345, -0.741, 1.469, -1.141, 0.097, -4.25, 0.0]

        # run and save the robot configurations 
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict_sim) 

    def calib_diode_laser_incident_dir(self):
        """calibration orientation with the laser direction (cutting laser)
        1. use the laser fiber tip to achieve system calibration 
        2. config: starts from the centers 
        """
        
        # get the robot configuration
        para_dict_sim = {}
        para_dict_sim["config_tip_ee_orientation_degree"] = [0.0, 0.0, 0.0]
        para_dict_sim["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 
        para_dict_sim["x_len"] = 0.005
        para_dict_sim["y_len"] = 0.005
        para_dict_sim["x_center"] = +0.008       # move to negative
        para_dict_sim["y_center"] = +0.002       # move to positive
        para_dict_sim["num_of_pts_per_line"] = 2
        para_dict_sim["z_height_scanning_grid"] = 0.0
        
        para_dict_sim["flag_save"] = "true"
        para_dict_sim["flag_vis"] = "false"
        para_dict_sim["path_config_save"] = "./data_calib/laser_diode_incident_dir_raster.npy"

        # laser configuration
        para_dict_sim["config_tumorid_klampt"] = [0.0, -4.345, -0.741, 1.469, -1.141, 0.097, -4.25, 0.0]

        # run and save the robot configurations 
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict_sim) 

    def calib_laser_tumorid_incident_dir(self):
        """calibration orientation with the laser direction
        1. use the laser fiber tip to achieve system calibration 
        2. config: starts from the centers 
        3. tumorid laser only (compare with the laser diode)
        """
        
        # get the robot configuration
        para_dict_sim = {}
        para_dict_sim["config_tip_ee_orientation_degree"] = [0.0, 0.0, 0.0]
        para_dict_sim["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 
        para_dict_sim["x_len"] = 0.006
        para_dict_sim["y_len"] = 0.006
        para_dict_sim["x_center"] = -0.120        # move to negative
        para_dict_sim["y_center"] = -0.032        # move to positive
        para_dict_sim["num_of_pts_per_line"] = 2
        para_dict_sim["z_height_scanning_grid"] = -0.02
        
        para_dict_sim["flag_save"] = "true"
        para_dict_sim["flag_vis"] = "true"
        para_dict_sim["path_config_save"] = "./data_calib/laser_incident_dir_raster.npy"

        # laser configuration
        para_dict_sim["config_tumorid_klampt"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.282, 0.0]

        # run and save the robot configurations 
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict_sim) 

    def calib_cutting_laser_axis_dir(self):
        """calibration orientation with the laser direction"""

        """
        input: given a 3d trajectory
        output: achieve the raster scanning"""
        
        # get the robot configuration
        para_dict_sim = {}
        para_dict_sim["config_tip_ee_orientation_degree"] = [0.0, 0.0, 0.0]
        para_dict_sim["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 
        para_dict_sim["x_len"] = 0.006
        para_dict_sim["y_len"] = 0.006
        para_dict_sim["x_center"] = +0.002       # move to negative
        para_dict_sim["y_center"] = +0.000       # move to positive
        para_dict_sim["num_of_pts_per_line"] = 3
        para_dict_sim["z_height_scanning_grid"] = 0.0
        
        para_dict_sim["flag_save"] = "true"
        para_dict_sim["flag_vis"] = "true"
        para_dict_sim["path_config_save"] = "./data_calib/laser_cutting_axis_orientation_raster_3.npy"

        # laser configuration
        para_dict_sim["config_tumorid_klampt"] = [0.0, -4.345, -0.741, 1.469, -1.141, 0.097, -4.25, 0.0]

        # run and save the robot configurations 
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict_sim) 

    def ik_general_laser_solver(self, flag_laser = [] ):
        """generalized ik solver with a 3d roi from the laser plane
            1. robot moving in a plane with fixed distance 
            2. tumorid -> laser-1. 
            3. laser diode -> laser-2. 
            return: robot trajectory (configurations)
        """ 

        # get the robot configuration
        input("check the input configurations")

        if flag_laser == "tumorid": 
            config_tip_ee_orientation_degree = [0.0, 0.0, 0.0]
            config_tip_ee_translation = [0.0, 0.0, 0.0] 
            x_len = 0.006
            y_len = 0.006
            x_center = -0.118                   # move to negative      
            y_center = -0.032                   # check the calibration scanning pattern
            num_of_pts_per_line = 3 
            z_height_scanning_grid = -0.02
            flag_save = "true"
            flag_vis = "false"
            path_config_save = "./database/data_ik/traj_ik_test_" + flag_laser + ".npy"
            config_tumorid_klampt = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.282, 0.0]
        elif flag_laser == "diode": 
            config_tip_ee_orientation_degree = [0.0, 0.0, 0.0]
            config_tip_ee_translation = [0.0, 0.0, 0.0] 
            x_len = 0.005
            y_len = 0.005
            x_center = 0.008                    # move to negative      
            y_center = 0.002                    # check the calibration scanning pattern
            num_of_pts_per_line = 2
            z_height_scanning_grid = 0.0
            flag_save = "true"
            flag_vis = "false"
            path_config_save = "./database/data_ik/traj_ik_test_" + flag_laser + ".npy"
            config_tumorid_klampt = [0.0, -4.345, -0.741, 1.469, -1.141, 0.097, -4.25, 0.0]
        elif flag_laser == "fiber":
            # fiber couple laser
            config_tip_ee_orientation_degree = [0.0, 0.0, 0.0]
            config_tip_ee_translation = [0.0, 0.0, 0.0] 
            x_len = 0.005
            y_len = 0.005
            x_center = 0.008                    # move to negative      
            y_center = 0.002                    # check the calibration scanning pattern
            num_of_pts_per_line = 2
            z_height_scanning_grid = 0.0
            flag_save = "true"
            flag_vis = "false"
            path_config_save = "./database/data_ik/traj_ik_test_" + flag_laser + ".npy"
            config_tumorid_klampt = [0.0, -4.345, -0.741, 1.469, -1.141, 0.097, -4.25, 0.0]

        # reference grids 
        para_dict_3d_grid = {}
        para_dict_3d_grid["x_len"] = x_len
        para_dict_3d_grid["y_len"] = y_len 
        para_dict_3d_grid["x_center"] = x_center
        para_dict_3d_grid["y_center"] = y_center
        para_dict_3d_grid["num_of_pts_per_line"] = num_of_pts_per_line
        para_dict_3d_grid["z_height_scanning_grid"] = z_height_scanning_grid
        para_dict_3d_grid["flag_vis"] = "false"
        pts_grid = self.robot_utility.pts_3d_scanning_grid(para_dict=para_dict_3d_grid)

        # TODO: fid checking only (referecne only)
        # trajectory from the ik solvers
        # vx_local = [1.0, 0.0, 0.0]
        # vy_local = [0.0, 1.0, 0.0]
        # traj_x = scipy.io.loadmat("./data_test/list_of_beta_x_opt.mat")["list_of_beta_x_opt"]
        # traj_y = scipy.io.loadmat("./data_test/list_of_beta_y_opt.mat")["list_of_beta_y_opt"]
        fid_pts_xy = scipy.io.loadmat("./database/data_ik/list_of_pts_robot_local_fid.mat")["list_of_pts_robot_local_fid"]
        fid_pts_xyz = fid_pts_xy
        fid_pts_xyz[:,2] = z_height_scanning_grid * 1000.0
        fid_pts_xyz = fid_pts_xyz / 1000.0
        fid_pts_xyz[:,0] += x_center
        fid_pts_xyz[:,1] += y_center
        
        # TODO: actual trajectory from the matlab IK and planning solver 
        # trajectory as the selected/segmented targets
        path_traj_pts_xy = "./database/data_ik/list_of_pts_robot_local_traj_" + flag_laser + ".mat"
        traj_pts_xy = scipy.io.loadmat(path_traj_pts_xy)["list_of_pts_robot_local_traj"]
        traj_pts_xyz = traj_pts_xy
        traj_pts_xyz[:,2] = z_height_scanning_grid * 1000.0
        traj_pts_xyz = traj_pts_xyz / 1000.0
        traj_pts_xyz[:,0] += x_center
        traj_pts_xyz[:,1] += y_center

        # vis 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts_grid[:,0], pts_grid[:,1], pts_grid[:,2], c = 'b', marker='o')
        ax.scatter(fid_pts_xyz[:,0], fid_pts_xyz[:,1], fid_pts_xyz[:,2], c = 'r', marker = 'o')
        ax.scatter(traj_pts_xyz[:,0], traj_pts_xyz[:,1], traj_pts_xyz[:,2], c = 'g', marker = 'o')

        # fid text 
        for idx_tmp in range(len(fid_pts_xyz[:,0])): 
            ax.text(fid_pts_xyz[idx_tmp,0], fid_pts_xyz[idx_tmp,1], fid_pts_xyz[idx_tmp,2], str(idx_tmp), fontsize=15)
        # grid text
        for idx_tmp in range(len(pts_grid[:,0])): 
            ax.text(pts_grid[idx_tmp,0] + 0.0005, pts_grid[idx_tmp,1] + 0.0005, pts_grid[idx_tmp,2] + 0.0005, str(idx_tmp), color = "b", fontsize=10)
        # traj text
        for idx_tmp in range(len(traj_pts_xyz[:,0])): 
            ax.text(traj_pts_xyz[idx_tmp,0] + 0.0005, traj_pts_xyz[idx_tmp,1] + 0.0005, traj_pts_xyz[idx_tmp,2] + 0.0005, str(idx_tmp), color = "g", fontsize=10)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

        # world robot
        path_robot_file = self.robot_utility.xmlfile
        res = self.robot_utility.world.readFile(path_robot_file)
        if not res:
            raise RuntimeError("unable to load model")
        self.robot = self.robot_utility.world.robot(0)
        ee_link = 7
        base_link = 0
        self.EELink = self.robot.link(ee_link)
        self.BaseLink = self.robot.link(base_link)
        self.robot.setConfig(config_tumorid_klampt)
        T_world = klampt.math.se3.identity()
        T_Base = self.BaseLink.getTransform()
        T_EE = self.EELink.getTransform()
        T_EE_init = T_EE

        # design the end-effector of the tip  
        angle_x_axis = config_tip_ee_orientation_degree[0] / 180.0 * math.pi
        angle_y_axis = config_tip_ee_orientation_degree[1] / 180.0 * math.pi
        angle_z_axis = config_tip_ee_orientation_degree[2] / 180.0 * math.pi
        rotation_tip_to_ee = klampt.math.so3.from_rpy([ angle_x_axis, angle_y_axis, angle_z_axis ]) 
        T_EE2Tip = (rotation_tip_to_ee, [0.0, 0.0, 0.0])
        T_tip_translation = [ config_tip_ee_translation[0], config_tip_ee_translation[1], config_tip_ee_translation[2] ]  
        # T_EE2Tip_zero = (klampt.math.so3.from_rpy([ 0.0, 0.0, 0.0 ]) , [0.0, 0.0, 0.0])
        T_EE2Tip_zero = ( rotation_tip_to_ee , [0.0, 0.0, 0.0])

        # # scanning grid
        # # TODO: simplify the model -> finished.
        # para_dict_3d_grid = {}
        # para_dict_3d_grid["x_len"] = x_len
        # para_dict_3d_grid["y_len"] = y_len 
        # para_dict_3d_grid["x_center"] = x_center
        # para_dict_3d_grid["y_center"] = y_center
        # para_dict_3d_grid["num_of_pts_per_line"] = num_of_pts_per_line
        # para_dict_3d_grid["z_height_scanning_grid"] = z_height_scanning_grid
        # para_dict_3d_grid["flag_vis"] = flag_vis
        # pts_grid = self.pts_3d_scanning_grid(para_dict=para_dict_3d_grid)

        # simulation
        # the first configuration is the home-configuration (skip) 
        # getWorldPosition of the {EE} end-effector position  
        # orientation with the end-effector
        # initialTransform: 3 x 1 position of the tip position
        q0 = config_tumorid_klampt
        Qs = [q0]
        pts_global_grid = []
        pts_tip_world = self.EELink.getWorldPosition(T_tip_translation)
        pts_use_sim = traj_pts_xyz
        # print("pts_use_sim shape = ", pts_use_sim.shape)
        # exit()

        # TODO: get the orientation of the {EE} 
        # vectors within the local frame 
        tform_ee = self.EELink.getTransform() 
        R_ee_in_world, t_ee_in_world = tform_ee
        vec_ee_in_world_x = R_ee_in_world[0:3]
        vec_ee_in_world_y = R_ee_in_world[3:6]
        vec_ee_in_world_z = R_ee_in_world[6:9]
        offset_global_1 = ( np.asarray([0.0, 0.0, 0.0]) * 0.0 ).tolist()
        offset_global_2 = ( np.asarray(vec_ee_in_world_x) * 0.05 ).tolist()
        offset_global_3 = ( np.asarray(vec_ee_in_world_y) * 0.05 ).tolist()

        # loop for simulation
        count_success = 0 
        scale_1 = 0.0
        scale_2 = scale_3 = 0.05
        for i in range(len(pts_use_sim)):

            # current point
            pos_in_local_grid = pts_use_sim[i]  

            print("i = ", i)

            # The local point definition
            # this is baesd on the reference link: self.EELink
            pos1 = [0.0, 0.0, 0.0]
            pos2 = [0.05, 0.0, 0.0]
            pos3 = [0.0, 0.05, 0.0]

            # TODO: fix the orientation
            local1 = klampt.math.se3.apply(T_EE2Tip_zero, pos1)
            local2 = klampt.math.se3.apply(T_EE2Tip_zero, pos2)
            local3 = klampt.math.se3.apply(T_EE2Tip_zero, pos3)

            # The global point definition
            # Define in {W}
            # pts_tip_world: point in the {EE} in {world}
            # pts_use_sim: point in the grid 
            global1 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_1 ))  
            global2 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_2 ))
            global3 = klampt.math.vectorops.add(pts_tip_world, klampt.math.vectorops.add(pos_in_local_grid, offset_global_3 ))
            pts_global_grid.append(global1)
            goal = klampt.model.ik.objective(self.EELink, local=[local1, local2, local3], world=[global1, global2, global3])

            # if ik.solve_nearby(goal, maxDeviation = 0.2, tol = 0.001, activeDofs = self.activeDoFs):
            if klampt.model.ik.solve(goal, tol=0.00001):
                print("IK solver success")
                print(local1, local2, local3, global1, global2, global3)
                goalConfig = self.robot.getConfig()
                self.robot.setConfig(self.robot.getConfig())
                # print(pos_target)
                # print(self.robot.getConfig())
                # raw_input()
                Qs.append(self.robot.getConfig())
                count_success += 1
            else:
                print("IK solver failure")
                print(pts_grid[i])
                goalConfig = self.robot.getConfig()
                Qs.append(self.robot.getConfig())

        if count_success == (len(Qs) - 1):
            print("Configuration success")
            input("check success robot configuration")
        else:
            print("IK failed")
            exit()
        
        # print("flag_vis = ", flag_vis)
        # exit()

        # save the robot configuration 
        if flag_save == "true": 
            Qs.pop(0)
            print("path_config_save = ", path_config_save)
            # np.save(path_config_save, Qs)
            print("saved the robot configuration trajectory")

        # Show the frame
        # Define the Trajectory
        if flag_vis == "true": 
            traj0 = klampt.model.trajectory.RobotTrajectory(self.robot, times=list(range(len(Qs))), milestones=Qs)
            vis.add("world", self.robot_utility.world)
            vis.show()
            # Add the points
            for idx, item in enumerate(pts_global_grid):
                vis.add(str(idx), item)
            while vis.shown():
                vis.lock()
                t = vis.animationTime()

                # Update the frames
                T_Base = self.BaseLink.getTransform()
                T_EE = self.EELink.getTransform()
                T_EE2Tip = klampt.math.se3.identity()
                (R, T) = T_EE2Tip
                # T_tip_translation = [0.0, 0.0, -0.1]
                T_EE2Tip = (rotation_tip_to_ee, T_tip_translation)
                T_Tip = klampt.math.se3.mul(T_EE, T_EE2Tip)

                # Show the frame
                vis.add("World Frame", T_world)
                vis.add("Base Frame", T_Base)
                vis.add("EE", T_EE)
                # vis.add("Tip", T_Tip)
                # vis.add("EE_init", T_EE_init)

                q = traj0.eval(t, endBehavior='loop')
                self.robot.setConfig(q)
                vis.unlock()
                time.sleep(0.01)
                print(int(t))

            # vis.spin(float('inf'))
            vis.show(False)
            vis.kill()

        return Qs

    def calib_laser_roi(self): 
        """
        input: given a 3d trajectory
        output: achieve the raster scanning"""
        
        # get the robot configuration
        para_dict_sim = {}
        para_dict_sim["config_tip_ee_orientation_degree"] = [0.0, 0.0, 0.0]
        para_dict_sim["config_tip_ee_translation"] = [0.0, 0.0, 0.0] 
        para_dict_sim["x_len"] = 0.009
        para_dict_sim["y_len"] = 0.009
        para_dict_sim["x_center"] = -0.119       # move to negative
        para_dict_sim["y_center"] = -0.033       # move to positive
        para_dict_sim["num_of_pts_per_line"] = 3 
        para_dict_sim["z_height_scanning_grid"] = -0.02
        
        para_dict_sim["flag_save"] = "true"
        para_dict_sim["flag_vis"] = "true"
        para_dict_sim["path_config_save"] = []

        # tumorid initialization
        para_dict_sim["config_tumorid_klampt"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.282, 0.0]

        # laser configuration

        # run and save the robot configurations 
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict_sim) 

    def surface_raster_scan_traj(self, path_ref = []): 
        """
        input: given a 3d trajectory
        output: achieve the raster scanning"""
        
        # get the robot configuration
        para_dict_sim = {}

        # path configuration 
        para_dict_sim["path_config_save"]                   = path_ref 
        para_dict_sim["config_tip_ee_orientation_degree"]   = [0.0, 0.0, 0.0]
        para_dict_sim["config_tip_ee_translation"]          = [0.0, 0.0, 0.0] 

        # example testing (mesh grids) unit: meters
        para_dict_sim["x_len"]                  =  0.013
        para_dict_sim["y_len"]                  =  0.013
        para_dict_sim["x_center"]               = -0.118       
        para_dict_sim["y_center"]               = -0.032      
        para_dict_sim["num_of_pts_per_line"]    = 8
        para_dict_sim["z_height_scanning_grid"] = -0.02
        
        # trajectory path folder 
        para_dict_sim["flag_save"] = "true"
        para_dict_sim["flag_vis"]  = "true"
    
        # tumorid initialization
        para_dict_sim["config_tumorid_klampt"] = [0.0, -4.424, -0.925, 1.833, -1.187, 0.099, -4.282, 0.0]

        # run and save the robot configurations 
        traj_robot_config = self.robot_simulation.simulation_scanning_plane(para_dict=para_dict_sim) 

class exp_main():

    def __init__(self):
        
        # define the class 
        self._test_class = robot_unit_test() 

    def scan_surface_exp(self): 

        # surface raster scanning
        path_ref = "./database/data_mice_official/surface_raster/surface_raster_scan_check.npy"
        self._test_class.surface_raster_scan_traj(path_ref=path_ref) 

    def data_collection_exp(self): 

        # histopathology verification (line-scan)
        self._test_class.planar_line_scan_histopathology() 

        # tumor resection verification (line-scan)
        self._test_class.planar_line_scan_resection() 

        # histopathology
        self._test_class.planar_raster_scan_histopathology()

        # resection exp degree raster scan
        self._test_class.planar_raster_scan_resection()

    def solver_inverse_kinematics_exp(self):

        # ik solver: tumorid
        self._test_class.ik_general_laser_solver(flag_laser="tumorid")

        # visible green laser diode scalpel 
        self._test_class.ik_general_laser_solver(flag_laser="diode")

        # fiber-coupled laser scalpel
        self._test_class.ik_general_laser_solver(flag_laser="fiber")

    def calibration_exp(self):
        
        # oct-to-laser axis calibration
        self._test_class.calib_cutting_laser_axis_dir() 

        # calibration roi 
        self._test_class.calib_laser_roi()
        
        # calibration -> cutting laser: laser incident orientation (with the cutting laser diode)
        self._test_class.calib_laser_tumorid_incident_dir()             # tumorid
        self._test_class.calib_diode_laser_incident_dir()               # diode 
        self._test_class.calib_fiber_laser_incident_dir()               # fiber

if __name__ == "__main__":

    exp_class = exp_main()

    """calibration experiments"""
    exp_class.calibration_exp() 

    """data colleciton platform"""
    exp_class.data_collection_exp()

    """tissue experiment (scanning surface)
        exvivo tissue study
        mice tumor study 
        system study
    """
    print("use following two functions to finish the robot planning for experiments")
    exp_class.scan_surface_exp()
    exp_class.solver_inverse_kinematics_exp()