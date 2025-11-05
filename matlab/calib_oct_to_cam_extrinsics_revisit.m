
clc; 
clear all; 
close all; 

%% start the process 

% step-1
path_main_global          = "./dataset/calibration/";
path_main_oct_to_cam      = strcat( path_main_global, "cam_to_oct/" ); 
path_main_cam_intrinsics  = strcat( path_main_global, "cam_intrinsics/" ); 
num_of_img                = 6; 

% camera-to-oct calibration
pts_fid_2d_left_all       = []; 
pts_fid_2d_right_all      = [];
pts_fid_3d_all            = [];

for idx_pose = 1 : num_of_img 
    
    % load the data 
    path_data            = strcat( path_main_oct_to_cam, "fid_label_pose_", num2str(idx_pose), ".mat" ); 
    load(path_data);  

    % left-2d:
    pts_fid_2d_left_all  = [pts_fid_2d_left_all; pts_fid_2d_left]; 

    % right-2d: 
    pts_fid_2d_right_all = [pts_fid_2d_right_all; pts_fid_2d_right]; 

    % oct-3d: 
    pts_fid_3d_all       = [pts_fid_3d_all; pts_fid_3d]; 

end

%% reprojection errors 

% define the pixel pitch 
pixel_pitch = 0.15;

% load the intrinsic calibration parameters 
path_left_cam_intrinsics  = load( strcat( path_main_cam_intrinsics, "cam_left_intrinsic.mat" ) );
cam_intrinsic_para_left   = path_left_cam_intrinsics.cameraParams_left;

path_right_cam_intrinsics = load( strcat( path_main_cam_intrinsics, "cam_right_intrinsic.mat" ) );
cam_intrinsic_para_right  = path_right_cam_intrinsics.cameraParams_right;

% extrinsics calibration (cam-left)
[ worldPose_left, worldVector_left ] = estimateWorldCameraPose( pts_fid_2d_left_all,  pts_fid_3d_all, cam_intrinsic_para_left.Intrinsics );
tform_cam_to_oct            = rigid3d( worldPose_left, worldVector_left );
rot_cam_to_oct              = worldPose_left'; 
trans_cam_to_oct            = worldVector_left; 
R_tform_inv                 = (worldPose_left')^(-1);
t_tform_inv                 = -R_tform_inv * worldVector_left';
tform_oct_to_cam            =  rigid3d( R_tform_inv', t_tform_inv' );
tform_oct_to_cam_left       = tform_oct_to_cam; 

% rpj errors
pts_fid_2d_left_from_oct    = worldToImage( cam_intrinsic_para_left.Intrinsics, tform_oct_to_cam_left , pts_fid_3d_all );
dis_left_cam                = pts_fid_2d_left_all - pts_fid_2d_left_from_oct; 
err_left_cam                = sqrt( dis_left_cam(:,1).^2 + dis_left_cam(:,2).^2 );
err_left_cam_in_mm          = err_left_cam .* pixel_pitch;

% extrinsics calibration (cam-right)
[ worldPose_right, worldVector_right ] = estimateWorldCameraPose( pts_fid_2d_right_all, pts_fid_3d_all, cam_intrinsic_para_right.Intrinsics );
tform_cam_to_oct = rigid3d( worldPose_right, worldVector_right );
rot_cam_to_oct = worldPose_right'; 
trans_cam_to_oct = worldVector_right; 
R_tform_inv = (worldPose_right')^(-1);
t_tform_inv = -R_tform_inv * worldVector_right';
tform_oct_to_cam =  rigid3d( R_tform_inv', t_tform_inv' );
tform_oct_to_cam_right = tform_oct_to_cam; 

% rpj errors
pts_fid_2d_right_from_oct = worldToImage( cam_intrinsic_para_right.Intrinsics, tform_oct_to_cam_right , pts_fid_3d_all );
dis_right_cam = pts_fid_2d_right_all - pts_fid_2d_right_from_oct; 
err_right_cam = sqrt( dis_right_cam(:,1).^2 + dis_right_cam(:,2).^2 );
err_right_cam_in_mm          = err_right_cam .* pixel_pitch;