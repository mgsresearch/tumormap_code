
clc; 
clear all; 
close all;

%% laser axis + laser dir fid

% global parameter calibration 
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
z_ref = 5.63;                                  
num_of_var = 2;                             
pts_org_ref = [0.0, 0.0, z_ref]';            
vec_laser_init_gt = [0.0, 0.0, -1.0]';       
pts_org_init_gt = pts_org_ref; 
num_pose_coordinate = 4; 
% unit checking case
pts_org_init_gt(3) = 10; 

% calibration
date_of_calib_dir       = "use_7";
date_of_calib_save      = "use_7";
idx_of_laser            = "fiber";
name_of_data_save       = strcat( "calib_laser_dir_res_", idx_of_laser, "_" ); 
path_calib_data         = "./dataset/calibration/";
path_model_data         = "./dataset/modeling/"; 

% load the tissue model 
pts_xyz_exp = load( strcat( path_model_data, "pts_xyz_exp.mat") ).pts_xyz_exp; 
pts_rgb_exp = load( strcat( path_model_data, "pts_rgb_exp.mat") ).pts_rgb_exp; 

% laser axis data
date_of_calib_axis          = "use_1";
name_of_data                = "calib_laser_axis_"; 
path_calib_laser_axis_tmp   = strcat( path_calib_data, name_of_data, date_of_calib_axis, ".mat" ); 
load(path_calib_laser_axis_tmp);
fid_3d                      = data_struct.fid_3d; 
list_of_fid_axis_3d_unflip  = data_struct.list_of_fid_axis_3d_unflip; 
vx                          = data_struct.vx; 
vy                          = data_struct.vy; 
vz                          = data_struct.vz; 
vx_vis                      = data_struct.vx_vis; 
vy_vis                      = data_struct.vy_vis; 
vz_vis                      = data_struct.vz_vis; 

% vx_axis = vx;
% vy_axis = vy; 

% laser direction data 
name_of_data                = "calib_laser_dir_fid_"; 
path_calib_laser_dir_tmp    = strcat( path_calib_data, name_of_data, idx_of_laser, "_", date_of_calib_dir, ".mat" ); 
load(path_calib_laser_dir_tmp); 

list_of_laser_dir_beta      = data_struct.list_of_laser_dir_beta; 
list_of_laser_dir_fid       = data_struct.list_of_laser_dir_fid;
list_of_laser_dir_pn        = data_struct.list_of_laser_dir_pn;
list_of_laser_dir_vn        = data_struct.list_of_laser_dir_vn;

figure(1);clf; 
scatter( pts_xyz_exp(:,1), pts_xyz_exp(:,2), [], pts_rgb_exp ./ 255.0 );
title("label a randomized trajectory manually");
h = drawfreehand; 
pts_tumor_2d_label = h.Position;

% develop the four fid searching 
num_of_pts_label  = size( pts_tumor_2d_label, 1 );
num_of_pts_search = 20;
idx_traj_select   = round( 1 : num_of_pts_label ./ num_of_pts_search : num_of_pts_label );
pts_tumor_2d_select = pts_tumor_2d_label(idx_traj_select,:);
pts_tumor_3d_all = [];
for idx_tumor = 1 : size( pts_tumor_2d_label, 1 )
    idx_knn = knnsearch( [pts_xyz_exp(:,1), pts_xyz_exp(:,2)], pts_tumor_2d_label(idx_tumor,:),  "K", 1 );
    pts_tumor_3d_all = [pts_tumor_3d_all; pts_xyz_exp(idx_knn,:) ];
end 

pts_target_final = pts_tumor_3d_all; 


%% 

figure(2);clf; 
% pcshow( list_of_laser_dir_fid, "r", "MarkerSize", 5000);
% hold on;
pcshow( pts_xyz_exp, pts_rgb_exp, "MarkerSize", 500 );
hold on;
pcshow( pts_target_final, "r", "MarkerSize", 2500 );

% figure configuration
xlim([-20, +20]);
ylim([-20, +20]);
zlim([-10, +10]); 
xlabel("X (unit: mm)"); 
ylabel("Y (unit: mm)");
zlabel("Z (unit: mm)");
title("mice 3d profile");

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

% test: fk model     
w_var_test  = [0.0, 0.0, 0.0, 0.0]; 
[ ~, ~ ]    = jac_v2_4dof_laser_to_oct( w_var_test ); 

% optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options = optimoptions('lsqnonlin', ...
                       'Display','iter', ...
                       'SpecifyObjectiveGradient', true, ...
                       'CheckGradients', true, ...
                       'FiniteDifferenceType', 'central');
% options.Algorithm = "levenberg-marquardt";

fun = @jac_v2_4dof_laser_to_oct;  
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
[ ~, ~ ]    = jac_v2_4dof_laser_to_oct( w_var_opt ); 
dis_err     = pts_list_predict - list_of_target_pts;
err_predict_to_gt = sqrt( dis_err(:,1).^2 + dis_err(:,2).^2 + dis_err(:,3).^2 );
rmse_err    = sqrt( mean(err_predict_to_gt.^2) );
mean_err    = mean(err_predict_to_gt); 
max_err     =  max(err_predict_to_gt);
mat_from_theta_vec_degree_opt = RfromTheta( [theta_x_opt, theta_y_opt, 0.0] ); 
vec_laser_opt = mat_from_theta_vec_degree_opt * vec_laser_init_gt;
vec_laser_opt = vec_laser_opt ./ norm( vec_laser_opt );

% first frame
pts_org_init_gt_ref_v1         = pts_org_init_gt'; % [ 0.0, 0.0, 4.0 ]; 
pts_org_opt_new_frame_v1_vis   = pts_org_init_gt_ref_v1; 
vec_laser_opt_new_frame_v1_vis = vec_laser_opt * 8.0; 

% second frame
% optimal and new frame
% this could be adjusted 
% refernece: pts_org_init_gt + ( alpha_x + list_of_beta_x(idx_tmp) ) * list_of_vx_calib' +  ( alpha_y + list_of_beta_y(idx_tmp) ) * list_of_vy_calib';
pts_org_init_gt_ref_v2         = pts_org_init_gt'; % [ 0.0, 0.0, 4.0 ]; 
pts_org_opt_new_frame_v2_vis   = pts_org_init_gt_ref_v2 + alpha_x_opt * list_of_vx_calib + alpha_y_opt * list_of_vy_calib;
vec_laser_opt_new_frame_v2_vis = vec_laser_opt * 8.0; 

% % vis the orientation
% figure(2);
% hold on; 
% pcshow( pts_org_opt_new_frame_v1_vis, "c", "MarkerSize", 5000 );
% hold on;
% quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vx_vis(1), vx_vis(2), vx_vis(3), "r", "LineWidth", 8); 
% hold on;
% quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vy_vis(1), vy_vis(2), vy_vis(3), "g", "LineWidth", 8); 
% hold on;
% quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vz_vis(1), vz_vis(2), vz_vis(3), "b", "LineWidth", 8); 
% hold on;
% % quiver3( pts_org_opt_new_frame_v1_vis(1), pts_org_opt_new_frame_v1_vis(2), pts_org_opt_new_frame_v1_vis(3), vec_laser_opt_new_frame_v1_vis(1), vec_laser_opt_new_frame_v1_vis(2), vec_laser_opt_new_frame_v1_vis(3), "c", "LineWidth", 8); 
% hold on;
% text( pts_org_opt_new_frame_v1_vis(1) + 0.5, pts_org_opt_new_frame_v1_vis(2) + 0.5, pts_org_opt_new_frame_v1_vis(3) + 0.5, "Laser-1", "Color", "k", "FontSize", 30 );

% vis the orientation
figure(2);
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
text( pts_org_opt_new_frame_v2_vis(1) + 0.5, pts_org_opt_new_frame_v2_vis(2) + 0.5, pts_org_opt_new_frame_v2_vis(3) + 0.5, "Fiber-laser", "Color", "k", "FontSize", 30 );

% show the laser trajectory 
for idx_pts_in_path = 1 : length( list_of_beta_x ) 

    % incidence point 
    pts_incidence_tmp = pts_org_init_gt + ( alpha_x_opt + list_of_beta_x(idx_pts_in_path) ) * list_of_vx_calib' +  ( alpha_y_opt + list_of_beta_y(idx_pts_in_path) ) * list_of_vy_calib';

    % target point 
    line_connect_start_to_end = [ pts_incidence_tmp'; list_of_laser_dir_fid(idx_pts_in_path, :) ]; 

    % show the vis 
    figure(2);
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

% return; 

%% kinematics module 

global cell_traj_tracker_only_for_vis;
global idx_traj_tracker_only_for_vis;
global theta_x_opt_fiber;
global theta_y_opt_fiber; 
global vec_laser_init_gt_fiber;
global num_of_var_fiber; 
global list_of_pn_ik_fiber;
global list_of_vn_ik_fiber;
global pts_org_init_gt_fiber;
global alpha_x_opt_fiber;
global alpha_y_opt_fiber;
global list_of_vx_calib_fiber;
global list_of_vy_calib_fiber;
global list_of_target_ik_pts_fiber;
global pts_list_predict_fiber;

idx_traj_tracker_only_for_vis = 1;

% ik-dof
num_of_var_tumorid = 2;
num_of_var_diode = 2;
num_of_var_fiber = 2; 

% trajectory as targets 
% 3D labelled map
pts_traj_robot_plane = pts_target_final; 
vec_laser_opt_fiber = vec_laser_opt;
theta_x_opt_fiber = theta_x_opt; 
theta_y_opt_fiber = theta_y_opt; 
vec_laser_init_gt_fiber  = vec_laser_init_gt; 
alpha_x_opt_fiber = alpha_x_opt;
alpha_y_opt_fiber = alpha_y_opt;
list_of_vx_calib_fiber = list_of_vx_calib; 
list_of_vy_calib_fiber = list_of_vy_calib; 
pts_org_init_gt_fiber = pts_org_init_gt; 

% summary of the IK solvers 
% use the fid as an IK targets 
list_of_target_ik_pts_fiber = [];
list_of_pn_ik_fiber = []; 
list_of_vn_ik_fiber = [];
list_of_ik_fid_gt_fiber = pts_traj_robot_plane; 

for idx_ik = 1 : size( list_of_ik_fid_gt_fiber, 1 ) 

    % surface constraints of the pn
    % [0.0, 0.0, pts(3)];
    array_pn_current_plane = [0.0, 0.0, list_of_ik_fid_gt_fiber( idx_ik, 3 ) ];
    list_of_pn_ik_fiber = [ list_of_pn_ik_fiber; array_pn_current_plane ]; 

    % surface constraints of the vn
    vn_current_plane = [0.0, 0.0, 1.0];
    list_of_vn_ik_fiber = [ list_of_vn_ik_fiber; vn_current_plane ]; 

    % target points 
    % get the ik points 
    pts_input_tmp = list_of_ik_fid_gt_fiber(idx_ik,:)';
    vec_input_tmp = vec_laser_opt_fiber; 
    pts_n_tmp = array_pn_current_plane'; 
    vec_n_tmp = vn_current_plane'; 
    [pts_fk_tmp, ~, ~, ~, ~] = MirrorModel_3D_test_submit( pts_input_tmp, vec_input_tmp, pts_n_tmp, vec_n_tmp );
    list_of_target_ik_pts_fiber = [list_of_target_ik_pts_fiber; pts_fk_tmp']; 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tumorid optimization 
% IK optimization solvers
% interior point method
options_opt = optimoptions('fmincon', ...
                           'SpecifyObjectiveGradient', true, ...
                           'CheckGradients', false, ...
                           'FiniteDifferenceType', 'central', ... 
                           'Display','iter', ...
                           'Algorithm', 'interior-point');    

fun = @jac_v2_4dof_laser_to_oct_ik_solver_fiber_vis_only_submit; 

var_opt_input = zeros([size(list_of_target_ik_pts_fiber, 1) * num_of_var_fiber, 1]);
var_min_scale = -180;
var_max_scale = +180;
lower_bound_range = ones(size(var_opt_input)) .* var_min_scale; 
upper_bound_range = ones(size(var_opt_input)) .* var_max_scale; 
nonlcon = [];
Aeq = []; 
beq = [];
[var_opt_ik, ~] = fmincon(fun, var_opt_input, [], [], Aeq, beq, lower_bound_range, upper_bound_range, nonlcon, options_opt);

% fiber
vx_robot_local = +[0.0, +1.0, 0.0]; 
vy_robot_local = +[-1.0, 0.0, 0.0];
cen_offset_x = 0.0;
cen_offset_y = 0.0;
list_of_beta_x_opt = []; 
list_of_beta_y_opt = [];
list_of_pts_robot_local = []; 
list_of_pts_robot_ref_plane = [];
num_of_joint = length(var_opt_ik) / num_of_var_fiber; 
for idx_tmp = 1 : num_of_joint

   % optimized coordinates 
   list_of_beta_x_opt(idx_tmp) = var_opt_ik(num_of_var_fiber * idx_tmp - (num_of_var_fiber - 1)); 
   list_of_beta_y_opt(idx_tmp) = var_opt_ik(num_of_var_fiber * idx_tmp - (num_of_var_fiber - 1) + 1); 

   % {local} to {robot} -> robot frame defined differently 
   pts_robot_local = (list_of_beta_x_opt(idx_tmp) + cen_offset_x) * vx_robot_local + ...
                     (list_of_beta_y_opt(idx_tmp) + cen_offset_y) * vy_robot_local; 
   list_of_pts_robot_local = [list_of_pts_robot_local; pts_robot_local]; 

   % this is for reference plane only (this is important). 
   pts_robot_local_tmp = pts_org_init_gt_fiber + ...
                         ( alpha_x_opt_fiber + list_of_beta_x_opt(idx_tmp) ) * list_of_vx_calib_fiber' + ...
                         ( alpha_y_opt_fiber + list_of_beta_y_opt(idx_tmp) ) * list_of_vy_calib_fiber';
   list_of_pts_robot_ref_plane = [list_of_pts_robot_ref_plane; pts_robot_local_tmp'];
end

% summary of the data
list_of_pts_robot_local_traj = list_of_pts_robot_local; 

%% vis the frame

figure(2);
hold on;

% show the laser trajectory 
for idx_pts_in_path = 1 : size( list_of_target_ik_pts_fiber, 1 ) 
   
    % decode the data
    idx_count = num_of_var_fiber * idx_pts_in_path - (num_of_var_fiber - 1);
    beta_x = var_opt_ik( idx_count); 
    beta_y = var_opt_ik( idx_count + 1); 

    % incidence point 
    pts_incidence_tmp = pts_org_init_gt + ( alpha_x_opt + beta_x ) * list_of_vx_calib' +  ( alpha_y_opt + beta_y ) * list_of_vy_calib';

    % target point 
    line_connect_start_to_end = [ pts_incidence_tmp'; list_of_target_ik_pts_fiber(idx_pts_in_path, :) ]; 

    % show the vis 
    figure(2);
    hold on; 
    pcshow( pts_incidence_tmp', "r", "MarkerSize", 2500 );
    hold on;
    quiver3( pts_incidence_tmp(1), pts_incidence_tmp(2), pts_incidence_tmp(3), vec_laser_opt_new_frame_v2_vis(1), vec_laser_opt_new_frame_v2_vis(2), vec_laser_opt_new_frame_v2_vis(3), "c", "LineWidth", 5); 
    hold on; 
    plot3( line_connect_start_to_end(:,1), line_connect_start_to_end(:,2), line_connect_start_to_end(:,3), "c", "LineWidth", 5 ); 

end 

set(gca,'XColor',[1 1 1],'YColor', [1 1 1],'ZColor', [1 1 1 ]);
set(gca, "FontSize", 20);
set(gca, "Color", "w"); 
set(gcf, "Color", "w"); 
