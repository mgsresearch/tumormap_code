
clc; 
clear all; 
close all;

%% settings

global list_of_target_pts; 
global list_of_pn; 
global list_of_vn; 
global list_of_beta_x;
global list_of_beta_y;
global vec_laser_init_gt;
global pts_org_init_gt;
global list_of_vx_calib; 
global list_of_vy_calib; 
global list_of_vz_calib;
global pts_list_predict;
global alpha_x_opt; 
global alpha_y_opt; 
global theta_x_opt; 
global theta_y_opt;
global list_of_target_ik_pts; 
global list_of_pn_ik; 
global list_of_vn_ik;

num_of_calib_board = 3;                        
gridStep_down = 0.20;                        
z_ref = 56.3 * cosd(40);                        
num_of_var = 2;                            
pts_org_ref = [0.0, 0.0, z_ref]';              
vec_laser_init_gt = [0.0, 0.0, -1.0]';         
pts_org_init_gt = pts_org_ref; 
num_pose_coordinate = 4; 

% develop the system calibration model s 
date_of_calib_dir   = "use_7";
date_of_calib_save  = "use_7";
idx_of_laser        = "fiber";
name_of_data_save   = strcat( "calib_laser_dir_res_", idx_of_laser, "_" ); 
path_save_data      = "./dataset/calibration/";

% laser axis data
date_of_calib_axis          = "use_1";
name_of_data                = "calib_laser_axis_"; 
path_calib_laser_axis_tmp   = strcat( path_save_data, name_of_data, date_of_calib_axis, ".mat" ); 
load(path_calib_laser_axis_tmp);

fid_3d                      = data_struct.fid_3d; 
list_of_fid_axis_3d_unflip  = data_struct.list_of_fid_axis_3d_unflip; 
vx                          = data_struct.vx; 
vy                          = data_struct.vy; 
vz                          = data_struct.vz; 
vx_vis                      = data_struct.vx_vis; 
vy_vis                      = data_struct.vy_vis; 
vz_vis                      = data_struct.vz_vis; 

% laser direction data 
name_of_data                = "calib_laser_dir_fid_"; 
path_calib_laser_dir_tmp    = strcat( path_save_data, name_of_data, idx_of_laser, "_", date_of_calib_dir, ".mat" ); 
load(path_calib_laser_dir_tmp); 

list_of_laser_dir_beta      = data_struct.list_of_laser_dir_beta; 
list_of_laser_dir_fid       = data_struct.list_of_laser_dir_fid;
list_of_laser_dir_pn        = data_struct.list_of_laser_dir_pn;
list_of_laser_dir_vn        = data_struct.list_of_laser_dir_vn;

%% optimization

list_of_calib_fid   = list_of_laser_dir_fid;
list_of_beta        = list_of_laser_dir_beta; 
list_of_target_pts  = [];
list_of_pn          = []; 
list_of_vn          = [];
list_of_beta_x      = []; 
list_of_beta_y      = []; 
list_of_vx_calib    = vx;
list_of_vy_calib    = vy;
list_of_vz_calib    = vz;

for idx_calib = 1 : length(list_of_calib_fid) 

    % 3D target points 
    list_of_target_pts  = [list_of_target_pts; list_of_calib_fid(idx_calib,:)]; 

    % surface constraints of the pn
    list_of_pn          = [list_of_pn; list_of_laser_dir_pn(idx_calib,:)]; 

    % surface constraints of the vn
    list_of_vn          = [list_of_vn; list_of_laser_dir_vn(idx_calib,:)]; 

    % beta_x and beta_y from the robot trajectory 
    % unit: mm 
    list_of_beta_x      = [list_of_beta_x; list_of_beta(idx_calib,1)]; 
    list_of_beta_y      = [list_of_beta_y; list_of_beta(idx_calib,2)]; 

end

% optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
options = optimoptions('lsqnonlin', ...
                       'Display','iter', ...
                       'SpecifyObjectiveGradient', true, ...
                       'CheckGradients', true, ...
                       'FiniteDifferenceType', 'central');
fun = @jac_v2_4dof_laser_to_oct_submit;

var_init = [ 0.0, 0.0, 0.0, 0.0 ];
lb = [-100, -100, -100, -100]; 
ub = [+100, +100, +100, +100]; 
[var_opt, resnorm2, residual2, exitflag2, output2] = lsqnonlin(fun, var_init, lb, ub, options);

% predicted coordinates from the given results 
alpha_x_opt = var_opt(1); 
alpha_y_opt = var_opt(2);
theta_x_opt = var_opt(3); 
theta_y_opt = var_opt(4); 

w_var_opt   = [ alpha_x_opt, alpha_y_opt, theta_x_opt, theta_y_opt ];
[ ~, ~ ]    = jac_v2_4dof_laser_to_oct_submit( w_var_opt ); 

dis_err     = pts_list_predict - list_of_target_pts;
err_predict_to_gt = sqrt( dis_err(:,1).^2 + dis_err(:,2).^2 + dis_err(:,3).^2 );
rmse_err    = sqrt( mean(err_predict_to_gt.^2) );
mean_err    = mean(err_predict_to_gt); 
max_err     =  max(err_predict_to_gt);
mat_from_theta_vec_degree_opt = RfromTheta_submit( [theta_x_opt, theta_y_opt, 0.0] ); 
vec_laser_opt = mat_from_theta_vec_degree_opt * vec_laser_init_gt;
vec_laser_opt = vec_laser_opt ./ norm( vec_laser_opt );

% first frame
pts_org_init_gt_ref_v1         = pts_org_init_gt';
pts_org_opt_new_frame_v1_vis   = pts_org_init_gt_ref_v1; 
vec_laser_opt_new_frame_v1_vis = vec_laser_opt * 8.0; 

% second frame
pts_org_init_gt_ref_v2         = pts_org_init_gt'; 
pts_org_opt_new_frame_v2_vis   = pts_org_init_gt_ref_v2 + alpha_x_opt * list_of_vx_calib + alpha_y_opt * list_of_vy_calib;
vec_laser_opt_new_frame_v2_vis = vec_laser_opt * 8.0; 

%% 

% vis the simulation engines 
figure(2);clf; 
pcshow( list_of_laser_dir_fid, "r", "MarkerSize", 5000);

% vis + label
% oct = world frame 
hold on;
quiver3( 0, 0, 0, 3, 0, 0, "r", "LineWidth", 8); 
hold on;
quiver3( 0, 0, 0, 0, 3, 0, "g", "LineWidth", 8); 
hold on; 
quiver3( 0, 0, 0, 0, 0, 3, "b", "LineWidth", 8); 
hold on; 
text( 0.0 + 0.5, 0.0 + 0.5, 0.0 + 0.5, "OCT", "Color", "b", "FontSize", 45 );

% local axis frame
hold on; 
pcshow( list_of_fid_axis_3d_unflip, "b", "MarkerSize", 5000);
hold on;
quiver3( list_of_fid_axis_3d_unflip(1,1), list_of_fid_axis_3d_unflip(1,2), list_of_fid_axis_3d_unflip(1,3), vx_vis(1), vx_vis(2), vx_vis(3), "r", "LineWidth", 8); 
hold on;
quiver3( list_of_fid_axis_3d_unflip(1,1), list_of_fid_axis_3d_unflip(1,2), list_of_fid_axis_3d_unflip(1,3), vy_vis(1), vy_vis(2), vy_vis(3), "g", "LineWidth", 8); 
hold on; 
quiver3( list_of_fid_axis_3d_unflip(1,1), list_of_fid_axis_3d_unflip(1,2), list_of_fid_axis_3d_unflip(1,3), vz_vis(1), vz_vis(2), vz_vis(3), "b", "LineWidth", 8); 
hold on;
text( list_of_fid_axis_3d_unflip(1,1), list_of_fid_axis_3d_unflip(1,2), list_of_fid_axis_3d_unflip(1,3), "Axis", "Color", "r", "FontSize", 30 );

% figure configuration
xlim([-20, +20]);
ylim([-20, +20]);
zlim([-10, +10]); 
xlabel("X (unit: mm)"); 
ylabel("Y (unit: mm)");
zlabel("Z (unit: mm)");

% vis the orientation
hold on; 
pcshow( pts_org_opt_new_frame_v1_vis, "c", "MarkerSize", 5000 );
hold on;
quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vx_vis(1), vx_vis(2), vx_vis(3), "r", "LineWidth", 8); 
hold on;
quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vy_vis(1), vy_vis(2), vy_vis(3), "g", "LineWidth", 8); 
hold on;
quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vz_vis(1), vz_vis(2), vz_vis(3), "b", "LineWidth", 8); 
hold on;
% quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vec_laser_opt_new_frame_v1_vis(1), vec_laser_opt_new_frame_v1_vis(2), vec_laser_opt_new_frame_v1_vis(3), "c", "LineWidth", 8); 
hold on;
% text( pts_org_opt_new_frame_v1_vis(1) + 0.5, pts_org_opt_new_frame_v1_vis(2) + 0.1, pts_org_opt_new_frame_v1_vis(3) + 0.5, "Laser-1", "Color", "k", "FontSize", 45 );

% vis the orientation
hold on; 
pcshow( pts_org_opt_new_frame_v2_vis, "c", "MarkerSize", 5000 );
hold on;
quiver3( pts_org_opt_new_frame_v2_vis(1), pts_org_opt_new_frame_v2_vis(2), pts_org_opt_new_frame_v2_vis(3), vx_vis(1), vx_vis(2), vx_vis(3), "r", "LineWidth", 8); 
hold on;
quiver3( pts_org_opt_new_frame_v2_vis(1), pts_org_opt_new_frame_v2_vis(2), pts_org_opt_new_frame_v2_vis(3), vy_vis(1), vy_vis(2), vy_vis(3), "g", "LineWidth", 8); 
hold on;
quiver3( pts_org_opt_new_frame_v2_vis(1), pts_org_opt_new_frame_v2_vis(2), pts_org_opt_new_frame_v2_vis(3), vz_vis(1), vz_vis(2), vz_vis(3), "b", "LineWidth", 8); 
hold on;
% quiver3( pts_org_opt_new_frame_v2_vis(1), pts_org_opt_new_frame_v2_vis(2), pts_org_opt_new_frame_v2_vis(3), vec_laser_opt_new_frame_v2_vis(1), vec_laser_opt_new_frame_v2_vis(2), vec_laser_opt_new_frame_v2_vis(3), "c", "LineWidth", 8); 
hold on;
% text( pts_org_opt_new_frame_v2_vis(1) + 0.5, pts_org_opt_new_frame_v2_vis(2) + 0.5, pts_org_opt_new_frame_v2_vis(3) + 0.5, "Laser-2", "Color", "k", "FontSize", 45 );

% show the laser trajectory 
for idx_pts_in_path = 1 : length( list_of_beta_x ) 
    
    % incidence point 
    pts_incidence_tmp = pts_org_init_gt + ( alpha_x_opt + list_of_beta_x(idx_pts_in_path) ) * list_of_vx_calib' +  ( alpha_y_opt + list_of_beta_y(idx_pts_in_path) ) * list_of_vy_calib';

    % target point 
    line_connect_start_to_end = [ pts_incidence_tmp'; list_of_laser_dir_fid(idx_pts_in_path, :) ]; 

    % show the vis 
    hold on; 
    pcshow( pts_incidence_tmp', "r", "MarkerSize", 2500 );
    % hold on;
    quiver3( pts_incidence_tmp(1), pts_incidence_tmp(2), pts_incidence_tmp(3), vec_laser_opt_new_frame_v2_vis(1), vec_laser_opt_new_frame_v2_vis(2), vec_laser_opt_new_frame_v2_vis(3), "r", "LineWidth", 5); 
    hold on; 
    % plot3( line_connect_start_to_end(:,1), line_connect_start_to_end(:,2), line_connect_start_to_end(:,3), "r", "LineWidth", 5 ); 

end 

set(gca,'XColor',[1 1 1],'YColor', [1 1 1],'ZColor', [1 1 1 ]);
set(gca, "FontSize", 20);
set(gca, "Color", "w"); 
set(gcf, "Color", "w"); 

%% 

% verify and visualization of the results 
figure(6);clf; 
pcshow(list_of_target_pts, "r", "MarkerSize", 2000);
hold on;
pcshow(pts_list_predict, "b", "MarkerSize", 1000);
xlim([-20, +20]);
ylim([-20, +20]);
zlim([-10, +10]); 
xlabel("X (unit: mm)"); 
ylabel("Y (unit: mm)");
zlabel("Z (unit: mm)");
title("3D point based offsets");
