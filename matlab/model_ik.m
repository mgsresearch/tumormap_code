
clc; 
clear all; 
close all;

global list_of_target_ik_pts_fiber; 
global num_of_var_fiber;
global list_of_target_ik_pts_tumorid; 
global list_of_pn_ik_tumorid; 
global list_of_vn_ik_tumorid;
global alpha_x_opt_tumorid; 
global alpha_y_opt_tumorid; 
global theta_x_opt_tumorid; 
global theta_y_opt_tumorid; 
global vec_laser_init_gt_tumorid;
global pts_org_init_gt_tumorid; 
global list_of_vx_calib_tumorid; 
global list_of_vy_calib_tumorid; 
global list_of_vz_calib_tumorid;
global pts_list_predict_tumorid; 
global num_of_var_tumorid; 

global list_of_target_ik_pts_fiber; 
global list_of_pn_ik_fiber; 
global list_of_vn_ik_fiber;
global alpha_x_opt_fiber; 
global alpha_y_opt_fiber; 
global theta_x_opt_fiber; 
global theta_y_opt_fiber; 
global vec_laser_init_gt_fiber;
global pts_org_init_gt_fiber; 
global list_of_vx_calib_fiber; 
global list_of_vy_calib_fiber; 
global list_of_vz_calib_fiber;
global pts_list_predict_fiber; 
global num_of_var_fiber; 
global num_of_var_tumorid; 

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

%% laser axis + laser dir fid

% save the calibration data 

% tumorid: 
% date_of_calib_dir = "use_2";
% idx_of_laser = "tumorid";

% diode: 
% date_of_calib_dir = "use_6";
% idx_of_laser = "diode"; 

% fiber
date_of_calib_dir = "use_3";
idx_of_laser = "fiber"; 

name_of_data_save = strcat( "calib_laser_dir_res_", idx_of_laser, "_" ); 
path_save_data = "./dataset/calibration/";

% laser axis data
date_of_calib_axis = "use_1";
name_of_data = "calib_laser_axis_"; 
path_calib_laser_axis_tmp = strcat( path_save_data, name_of_data, date_of_calib_axis, ".mat" ); 
load(path_calib_laser_axis_tmp); 

fid_3d = data_struct.fid_3d; 
list_of_fid_axis_3d_unflip = data_struct.list_of_fid_axis_3d_unflip; 
vx = data_struct.vx; 
vy = data_struct.vy; 
vz = data_struct.vz; 
vx_vis = data_struct.vx_vis; 
vy_vis = data_struct.vy_vis; 
vz_vis = data_struct.vz_vis;  

% laser direction data 
name_of_data = "calib_laser_dir_fid_"; 
path_calib_laser_dir_tmp = strcat( path_save_data, name_of_data, idx_of_laser, "_", date_of_calib_dir, ".mat" ); 
load(path_calib_laser_dir_tmp); 

list_of_laser_dir_beta = data_struct.list_of_laser_dir_beta; 
list_of_laser_dir_fid = data_struct.list_of_laser_dir_fid;
list_of_laser_dir_pn = data_struct.list_of_laser_dir_pn;
list_of_laser_dir_vn = data_struct.list_of_laser_dir_vn;

num_of_calib_board = 3;                         
gridStep_down = 0.20;                           
z_ref = 56.3 * cosd(40);                       
num_of_var = 2;                                 
pts_org_ref = [0.0, 0.0, z_ref]';               
vec_laser_init_gt = [0.0, 0.0, -1.0]';          
pts_org_init_gt = pts_org_ref; 

%% optimization
% estimation of the optimal laser incidence orientation 

list_of_calib_fid = list_of_laser_dir_fid;
list_of_beta = list_of_laser_dir_beta; 
list_of_target_pts = [];
list_of_pn = []; 
list_of_vn = [];
list_of_beta_x = []; 
list_of_beta_y = []; 
list_of_vx_calib = vx;
list_of_vy_calib = vy;
list_of_vz_calib = vz;

for idx_calib = 1 : length(list_of_calib_fid) 

    % 3D target points 
    list_of_target_pts = [list_of_target_pts; list_of_calib_fid(idx_calib,:)]; 

    % surface constraints of the pn
    list_of_pn = [list_of_pn; list_of_laser_dir_pn(idx_calib,:)]; 

    % surface constraints of the vn
    list_of_vn = [list_of_vn; list_of_laser_dir_vn(idx_calib,:)]; 

    % beta_x and beta_y from the robot trajectory 
    % unit: mm 
    list_of_beta_x = [list_of_beta_x; list_of_beta(idx_calib,1)]; 
    list_of_beta_y = [list_of_beta_y; list_of_beta(idx_calib,2)]; 

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

w_var_opt = [ alpha_x_opt, alpha_y_opt, theta_x_opt, theta_y_opt ];
[ ~, ~ ] = jac_v2_4dof_laser_to_oct_submit( w_var_opt ); 

dis_err = pts_list_predict - list_of_target_pts;
err_predict_to_gt = sqrt( dis_err(:,1).^2 + dis_err(:,2).^2 + dis_err(:,3).^2 );
rmse_err = sqrt( mean(err_predict_to_gt.^2) );
mean_err = mean(err_predict_to_gt); 
max_err =  max(err_predict_to_gt);
mat_from_theta_vec_degree_opt = RfromTheta_submit( [theta_x_opt, theta_y_opt, 0.0] ); 
vec_laser_opt = mat_from_theta_vec_degree_opt * vec_laser_init_gt;
vec_laser_opt = vec_laser_opt ./ norm( vec_laser_opt );

%% summarize the result

data_calib_res = [];
data_calib_res.vec_laser_init_gt = vec_laser_init_gt;
data_calib_res.pts_org_init_gt = pts_org_init_gt; 
data_calib_res.list_of_vx_calib = list_of_vx_calib;  
data_calib_res.list_of_vy_calib = list_of_vy_calib;
data_calib_res.list_of_vz_calib = list_of_vz_calib; 
data_calib_res.alpha_x_opt = alpha_x_opt; 
data_calib_res.alpha_y_opt = alpha_y_opt; 
data_calib_res.theta_x_opt = theta_x_opt; 
data_calib_res.theta_y_opt = theta_y_opt; 
data_calib_res.list_of_beta_x = list_of_beta_x; 
data_calib_res.list_of_beta_y = list_of_beta_y; 

%% test the ik system 

% develop a 3D meshgrid
thres_dis_edge = 2.5; 

num_of_pts  = 10; 
x_min_roi   = min( list_of_target_pts(:,1) ) - thres_dis_edge;
x_max_roi   = max( list_of_target_pts(:,1) ) + thres_dis_edge;
line_x_traj = x_min_roi : ( x_max_roi - x_min_roi ) ./ num_of_pts : x_max_roi;

y_min_roi  = min( list_of_target_pts(:,2) ) - thres_dis_edge;
y_max_roi  = max( list_of_target_pts(:,2) ) + thres_dis_edge;
line_y_traj = y_min_roi : ( y_max_roi - y_min_roi ) ./ num_of_pts : y_max_roi;

z_min_roi  = min( list_of_target_pts(:,3) ) - thres_dis_edge ;
z_max_roi  = max( list_of_target_pts(:,3) ) + thres_dis_edge;
line_z_traj = z_min_roi : ( z_max_roi - z_min_roi ) ./ num_of_pts : z_max_roi ;

[ x_traj_mesh, y_traj_mesh, z_traj_mesh ] = meshgrid( line_x_traj, line_y_traj, line_z_traj );

% 
pts_mesh_test = [x_traj_mesh(:), y_traj_mesh(:), z_traj_mesh(:)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tumorid optimization 
% IK optimization solvers
% interior point method

% preparation of the IK 
% ik-dof
num_of_var_tumorid          = 2;
num_of_var_diode            = 2;
list_of_target_ik_pts_fiber = [];
list_of_pn_ik_fiber         = []; 
list_of_vn_ik_fiber         = [];
list_of_ik_fid_gt_fiber     = pts_mesh_test; 

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
    vec_input_tmp = vec_laser_opt; 
    pts_n_tmp = array_pn_current_plane'; 
    vec_n_tmp = vn_current_plane'; 
    [pts_fk_tmp, ~, ~, ~, ~] = MirrorModel_3D_test_submit( pts_input_tmp, vec_input_tmp, pts_n_tmp, vec_n_tmp );
    list_of_target_ik_pts_fiber = [list_of_target_ik_pts_fiber; pts_fk_tmp']; 

end


% fiber
vec_laser_init_gt_fiber = data_calib_res.vec_laser_init_gt;
pts_org_init_gt_fiber   = data_calib_res.pts_org_init_gt; 
list_of_vx_calib_fiber  = data_calib_res.list_of_vx_calib; 
list_of_vy_calib_fiber  = data_calib_res.list_of_vy_calib;
list_of_vz_calib_fiber  = data_calib_res.list_of_vz_calib;
alpha_x_opt_fiber       = data_calib_res.alpha_x_opt;
alpha_y_opt_fiber       = data_calib_res.alpha_y_opt;
theta_x_opt_fiber       = data_calib_res.theta_x_opt;
theta_y_opt_fiber       = data_calib_res.theta_y_opt;

% list_of_target_ik_pts_fiber = pts_mesh_test; 
num_of_var_fiber = 2; 

options_opt = optimoptions('fmincon', ...
                           'SpecifyObjectiveGradient', true, ...
                           'CheckGradients', false, ...
                           'FiniteDifferenceType', 'central', ... 
                           'Display','iter', ...
                           'Algorithm', 'interior-point');    

fun = @jac_v2_4dof_laser_to_oct_ik_solver_fiber_submit; 

var_opt_input = zeros([size(list_of_target_ik_pts_fiber, 1) * num_of_var_fiber, 1]);
var_min_scale = -180;
var_max_scale = +180;
lower_bound_range = ones(size(var_opt_input)) .* var_min_scale; 
upper_bound_range = ones(size(var_opt_input)) .* var_max_scale; 
nonlcon = [];
Aeq = []; 
beq = [];
[var_opt_ik, ~] = fmincon(fun, var_opt_input, [], [], Aeq, beq, lower_bound_range, upper_bound_range, nonlcon, options_opt);

pts_list_predict_fiber = [];
[~, ~] = fun( var_opt_ik );

%% decode the dataset

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

% rpj error
dis_rpj = pts_list_predict_fiber - pts_mesh_test; 
rpj_err = sqrt( dis_rpj(:,1).^2 + dis_rpj(:,2).^2 + dis_rpj(:,3).^2 );

max( rpj_err )
% max( pts_list_predict_fiber )

% show the 3D heatmap 
figure(6);clf;  
pcshow( pts_mesh_test,rpj_err, "MarkerSize", 3000 );
colorbar;
c = colorbar;
c.Label.String = 'Error (unit:mm)';
set(gca, "FontSize", 50);
c.Label.FontSize = 50;
set(gca, "Color", "w");
set(gcf, "Color", "w");
title("3D error map for ik solvers");
