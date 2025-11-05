
clc; 
clear all; 
close all;

%% calibration module 

% path 
path_global_base = "./dataset/calibration/";

% camera intrinsics + extrinsics 
date_of_cam_in_and_extrin = "use_4"; 
path_cam_in_and_extrin_base = path_global_base;
path_cam_calib_extrin_file = strcat( path_cam_in_and_extrin_base, "calib_cam_to_oct_in_and_extrin_", date_of_cam_in_and_extrin, ".mat" ); 
load( path_cam_calib_extrin_file ); 
data_cam_in_and_extrin = data_calib_summary; 
pts_fid_3d_all = data_calib_summary.pts_fid_3d_all;
pts_fid_2d_left_all = data_calib_summary.pts_fid_2d_left_all;  
cam_intrinsic_para_left = data_calib_summary.cam_intrinsic_para_left; 
worldPose_left = data_calib_summary.worldPose_left; 
worldVector_left = data_calib_summary.worldVector_left;
tform_oct_to_cam_left = data_calib_summary.tform_oct_to_cam_left; 
pts_fid_2d_right_all = data_calib_summary.pts_fid_2d_right_all; 
cam_intrinsic_para_right = data_calib_summary.cam_intrinsic_para_right; 
worldPose_right = data_calib_summary.worldPose_right;
worldVector_right = data_calib_summary.worldVector_right;
tform_oct_to_cam_right = data_calib_summary.tform_oct_to_cam_right;

% tumorid config 
idx_of_laser = "tumorid"; 
name_of_data = "calib_laser_dir_res_"; 
date_of_calib_dir = "use_5";
path_calib_res = strcat( path_global_base, name_of_data, idx_of_laser, "_", date_of_calib_dir, ".mat" ); 
data_calib_res = [];
load(path_calib_res);
calib_res_tumorid = data_calib_res; 
vec_laser_init_gt_tumorid = calib_res_tumorid.vec_laser_init_gt; 
pts_org_init_gt_tumorid = calib_res_tumorid.pts_org_init_gt; 
list_of_vx_calib_tumorid = calib_res_tumorid.list_of_vx_calib; 
list_of_vy_calib_tumorid = calib_res_tumorid.list_of_vy_calib;
list_of_vz_calib_tumorid = calib_res_tumorid.list_of_vz_calib;
alpha_x_opt_tumorid = calib_res_tumorid.alpha_x_opt;  
alpha_y_opt_tumorid = calib_res_tumorid.alpha_y_opt; 
theta_x_opt_tumorid = calib_res_tumorid.theta_x_opt; 
theta_y_opt_tumorid = calib_res_tumorid.theta_y_opt; 
list_of_beta_x_tumorid = calib_res_tumorid.list_of_beta_x;
list_of_beta_y_tumorid = calib_res_tumorid.list_of_beta_y;
mat_from_theta_vec_degree_opt_tumorid = RfromTheta( [theta_x_opt_tumorid, theta_y_opt_tumorid, 0.0] ); 
vec_laser_opt_tumorid = mat_from_theta_vec_degree_opt_tumorid * vec_laser_init_gt_tumorid;
vec_laser_opt_tumorid = vec_laser_opt_tumorid ./ norm( vec_laser_opt_tumorid );
vec_laser_opt_tumorid_vis = vec_laser_opt_tumorid .* 10; 

% fiber config
idx_of_laser = "diode"; 
name_of_data = "calib_laser_dir_res_"; 
date_of_calib_dir = "use_6";
path_calib_res = strcat( path_global_base, name_of_data, idx_of_laser, "_", date_of_calib_dir, ".mat" ); 
data_calib_res = []; 
load(path_calib_res);
calib_res_fiber = data_calib_res; 
vec_laser_init_gt_fiber = calib_res_fiber.vec_laser_init_gt; 
pts_org_init_gt_fiber = calib_res_fiber.pts_org_init_gt; 
list_of_vx_calib_fiber = calib_res_fiber.list_of_vx_calib; 
list_of_vy_calib_fiber = calib_res_fiber.list_of_vy_calib;
list_of_vz_calib_fiber = calib_res_fiber.list_of_vz_calib;
alpha_x_opt_fiber = calib_res_fiber.alpha_x_opt;  
alpha_y_opt_fiber = calib_res_fiber.alpha_y_opt; 
theta_x_opt_fiber = calib_res_fiber.theta_x_opt; 
theta_y_opt_fiber = calib_res_fiber.theta_y_opt; 
mat_from_theta_vec_degree_opt_fiber = RfromTheta( [theta_x_opt_fiber, theta_y_opt_fiber, 0.0] ); 
vec_laser_opt_fiber = mat_from_theta_vec_degree_opt_fiber * vec_laser_init_gt_fiber;
vec_laser_opt_fiber = vec_laser_opt_fiber ./ norm( vec_laser_opt_fiber );
vec_laser_opt_fiber_vis = vec_laser_opt_fiber .* 10; 

% camera 
data_struct_calib_module.pts_fid_3d_all = pts_fid_3d_all; 
data_struct_calib_module.pts_fid_2d_left_all = pts_fid_2d_left_all; 
data_struct_calib_module.cam_intrinsic_para_left = cam_intrinsic_para_left;
data_struct_calib_module.worldPose_left = worldPose_left;
data_struct_calib_module.worldVector_left = worldVector_left; 
data_struct_calib_module.tform_oct_to_cam_left = tform_oct_to_cam_left;
data_struct_calib_module.pts_fid_2d_right_all = pts_fid_2d_right_all; 
data_struct_calib_module.cam_intrinsic_para_right = cam_intrinsic_para_right;
data_struct_calib_module.worldPose_right = worldPose_right;
data_struct_calib_module.worldVector_right = worldVector_right;
data_struct_calib_module.tform_oct_to_cam_right = tform_oct_to_cam_right; 

% tumorid
data_struct_calib_module.vec_laser_init_gt_tumorid = vec_laser_init_gt_tumorid; 
data_struct_calib_module.pts_org_init_gt_tumorid = pts_org_init_gt_tumorid; 
data_struct_calib_module.list_of_vx_calib_tumorid = list_of_vx_calib_tumorid; 
data_struct_calib_module.list_of_vy_calib_tumorid = list_of_vy_calib_tumorid; 
data_struct_calib_module.list_of_vz_calib_tumorid = list_of_vz_calib_tumorid; 
data_struct_calib_module.alpha_x_opt_tumorid = alpha_x_opt_tumorid; 
data_struct_calib_module.alpha_y_opt_tumorid = alpha_y_opt_tumorid; 
data_struct_calib_module.theta_x_opt_tumorid = theta_x_opt_tumorid; 
data_struct_calib_module.theta_y_opt_tumorid = theta_y_opt_tumorid; 
data_struct_calib_module.list_of_beta_x_tumorid = list_of_beta_x_tumorid; 
data_struct_calib_module.list_of_beta_y_tumorid = list_of_beta_y_tumorid; 
data_struct_calib_module.vec_laser_opt_tumorid = vec_laser_opt_tumorid; 
data_struct_calib_module.vec_laser_opt_tumorid_vis = vec_laser_opt_tumorid_vis; 

% fiber
data_struct_calib_module.vec_laser_init_gt_fiber = vec_laser_init_gt_fiber; 
data_struct_calib_module.pts_org_init_gt_fiber = pts_org_init_gt_fiber; 
data_struct_calib_module.list_of_vx_calib_fiber = list_of_vx_calib_fiber; 
data_struct_calib_module.list_of_vy_calib_fiber = list_of_vy_calib_fiber; 
data_struct_calib_module.list_of_vz_calib_fiber = list_of_vz_calib_fiber;
data_struct_calib_module.alpha_x_opt_fiber = alpha_x_opt_fiber;
data_struct_calib_module.alpha_y_opt_fiber = alpha_y_opt_fiber; 
data_struct_calib_module.theta_x_opt_fiber = theta_x_opt_fiber;
data_struct_calib_module.theta_y_opt_fiber = theta_y_opt_fiber; 
data_struct_calib_module.vec_laser_opt_fiber = vec_laser_opt_fiber;
data_struct_calib_module.vec_laser_opt_fiber_vis = vec_laser_opt_fiber_vis; 

%% vis the model at all 

% reference point cloud object 
figure(1);clf; 
pcshow(pts_fid_3d_all, "r", "MarkerSize", 500); 
hold on; 

% camera obj 
len_scale_world_cam = 20; 
pts_org_zero = [0.0, 0.0, 0.0];
x_axis_ref = [1.0, 0.0, 0.0] * len_scale_world_cam; 
y_axis_ref = [0.0, 1.0, 0.0] * len_scale_world_cam; 
z_axis_ref = [0.0, 0.0, 1.0] * len_scale_world_cam; 

% oct = world 
figure(1);
hold on; 
quiver3(pts_org_zero(1), pts_org_zero(2), pts_org_zero(3), x_axis_ref(1), x_axis_ref(2), x_axis_ref(3), "r", "LineWidth", 3); 
hold on; 
quiver3(pts_org_zero(1), pts_org_zero(2), pts_org_zero(3), y_axis_ref(1), y_axis_ref(2), y_axis_ref(3), "g", "LineWidth", 3); 
hold on;
quiver3(pts_org_zero(1), pts_org_zero(2), pts_org_zero(3), z_axis_ref(1), z_axis_ref(2), z_axis_ref(3), "b", "LineWidth", 3); 
hold on; 
text( pts_org_zero(1) + 5.0, pts_org_zero(2) + 5.0, pts_org_zero(3) + 5.0, "oct", "FontSize", 20, "Color", "r"); 

pts_org_left_cam = worldVector_left; 
len_scale_left_cam = 1; 
x_axis_left_cam = worldPose_left * x_axis_ref' * len_scale_left_cam;
y_axis_left_cam = worldPose_left * y_axis_ref' * len_scale_left_cam;
z_axis_left_cam = worldPose_left * z_axis_ref' * len_scale_left_cam;

% left-cam 
figure(1);
hold on; 
quiver3(pts_org_left_cam(1), pts_org_left_cam(2), pts_org_left_cam(3), x_axis_left_cam(1), x_axis_left_cam(2), x_axis_left_cam(3), "r", "LineWidth", 3); 
hold on; 
quiver3(pts_org_left_cam(1), pts_org_left_cam(2), pts_org_left_cam(3), y_axis_left_cam(1), y_axis_left_cam(2), y_axis_left_cam(3), "g", "LineWidth", 3); 
hold on;
quiver3(pts_org_left_cam(1), pts_org_left_cam(2), pts_org_left_cam(3), z_axis_left_cam(1), z_axis_left_cam(2), z_axis_left_cam(3), "b", "LineWidth", 3); 
hold on; 
text( pts_org_left_cam(1) + 5.0, pts_org_left_cam(2) + 5.0, pts_org_left_cam(3) + 5.0, "Left-cam", "FontSize", 20, "Color", "r"); 

pts_org_right_cam = worldVector_right; 
len_scale_right_cam = 1; 
x_axis_right_cam = worldPose_right * x_axis_ref' * len_scale_right_cam;
y_axis_right_cam = worldPose_right * y_axis_ref' * len_scale_right_cam;
z_axis_right_cam = worldPose_right * z_axis_ref' * len_scale_right_cam;

% right-cam 
figure(1);
hold on; 
quiver3(pts_org_right_cam(1), pts_org_right_cam(2), pts_org_right_cam(3), x_axis_right_cam(1), x_axis_right_cam(2), x_axis_right_cam(3), "r", "LineWidth", 3); 
hold on; 
quiver3(pts_org_right_cam(1), pts_org_right_cam(2), pts_org_right_cam(3), y_axis_right_cam(1), y_axis_right_cam(2), y_axis_right_cam(3), "g", "LineWidth", 3); 
hold on;
quiver3(pts_org_right_cam(1), pts_org_right_cam(2), pts_org_right_cam(3), z_axis_right_cam(1), z_axis_right_cam(2), z_axis_right_cam(3), "b", "LineWidth", 3); 
hold on; 
text( pts_org_right_cam(1) + 5.0, pts_org_right_cam(2) + 5.0, pts_org_right_cam(3) + 5.0, "Right-cam", "FontSize", 20, "Color", "r"); 

% tumorid
offset_tumorid_x_axis = -80.0; 
offset_tumorid_y_axis = -0.0;
pts_org_init_gt_tumorid_vis = [pts_org_init_gt_tumorid(1) + alpha_x_opt_tumorid .* 10 + offset_tumorid_x_axis, ...
                               pts_org_init_gt_tumorid(2) + alpha_y_opt_tumorid .* 10 + offset_tumorid_y_axis, ...
                               pts_org_init_gt_tumorid(3) + 50.0];
figure(1);
hold on;
pcshow(pts_org_init_gt_tumorid_vis, "r", "MarkerSize", 1000); 
hold on; 
quiver3( pts_org_init_gt_tumorid_vis(1), pts_org_init_gt_tumorid_vis(2), pts_org_init_gt_tumorid_vis(3), ...
         vec_laser_opt_tumorid_vis(1), vec_laser_opt_tumorid_vis(2), vec_laser_opt_tumorid_vis(3), "r", "LineWidth", 3); 

% fiber couple laser 
offset_fiber_x_axis = -80.0; 
offset_fiber_y_axis = -0.0;
pts_org_init_gt_fiber_vis = [pts_org_init_gt_fiber(1) + alpha_x_opt_fiber .* 10 + offset_fiber_x_axis, ...
                             pts_org_init_gt_fiber(2) + alpha_y_opt_fiber .* 10 + offset_fiber_y_axis, ...
                             pts_org_init_gt_fiber(3) + 50.0];
figure(1);
hold on;
pcshow(pts_org_init_gt_fiber_vis, "b", "MarkerSize", 1000); 
hold on; 
quiver3(pts_org_init_gt_fiber_vis(1), pts_org_init_gt_fiber_vis(2), pts_org_init_gt_fiber_vis(3), ...
        vec_laser_opt_fiber_vis(1), vec_laser_opt_fiber_vis(2), vec_laser_opt_fiber_vis(3), "b", "LineWidth", 3); 

figure(1);
hold on; 
xlabel("X unit:mm "); 
ylabel("Y unit:mm ");
zlabel("Z unit:mm "); 

set(gca,'CameraPosition',...
    [-599.026392489744 -729.966613292507 1268.16287238294],'CameraUpVector',...
    [0.542430807941069 0.571602804748391 0.615661475325663],'CameraViewAngle',...
    9.20630833093563,'Color',[1 1 1],'DataAspectRatio',[1 1 1],'Projection', 'perspective', ...
    'XColor',[1 1 1], 'YColor',[1 1 1], 'ZColor',[1 1 1]);
set(gca, "Color", "w"); 
set(gcf, "Color", "w"); 