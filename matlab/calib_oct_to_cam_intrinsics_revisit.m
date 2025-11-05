

clc;
clear all;
close all; 

% camera intrinsics calibration
cameraCalibrator;

%% save the data 

% left camera
idx_of_type = "left"; 
name_of_file = strcat( path_calib_cam_intrinsics_base, "cam_", idx_of_type, "_intrinsic_", date_of_data, ".mat" ); 
% save( name_of_file, "cameraParams_left" ); 

% right camera 
idx_of_type = "right"; 
name_of_file = strcat( path_calib_cam_intrinsics_base, "cam_", idx_of_type, "_intrinsic_", date_of_data, ".mat" ); 
% save( name_of_file, "cameraParams_right" ); 

