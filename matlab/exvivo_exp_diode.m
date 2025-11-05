
clc; 
clear all; 
close all; 

%% global 

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

%% system setting and configurations 

% fiber variables
num_of_var_fiber = 2;

% setting
idx_exvivo_name                      = "diode_cr_2"; 
path_data_unique                     = "exvivo/diode/";
path_global_all                      = "./dataset/"; 
path_global_calib                    = strcat( path_global_all, "calibration/" ); 
path_exvivo_main                     = strcat( path_global_all, path_data_unique );
path_save_img_global                 = strcat( path_global_all, "result_summary/" ); 
path_tissue_information              = path_exvivo_main; 
path_img_spot_centroid_left          = strcat( path_exvivo_main, "img_base_laser_spot_centroid_overlap.mat"); 

% settings + configurations
is_show_axis                         = "false"; 

% calibration summary
date_of_file                         = "use_6";  
path_save_data_base                  = path_global_calib;
path_calib_module_file               = strcat(path_save_data_base, "data_struct_calib_module_", date_of_file, ".mat"); 
data_struct_calib_module             = load( path_calib_module_file ).data_struct_calib_module; 

% Tumorid: 2d laser spots + reference images
label_center_spot_left               = readNPY( strcat(path_tissue_information, "label_center_spot_left.npy") ); 
label_center_spot_right              = readNPY( strcat(path_tissue_information, "label_center_spot_right.npy") ); 

img_base_left                        = load( strcat(path_tissue_information, "img_base_left.mat") ).img_base_left;
img_base_right                       = load( strcat(path_tissue_information, "img_base_right.mat") ).img_base_right;

% post-scanning 
img_base_left_post                   = load( strcat(path_tissue_information, "img_base_left.mat") ).img_base_left;
img_base_right_post                  = load( strcat(path_tissue_information, "img_base_right.mat") ).img_base_right;

% tumorid 
wavelength_npy                       = readNPY( strcat(path_tissue_information, "npy_of_wavelength.npy") ); 
spectral_npy                         = readNPY( strcat(path_tissue_information, "npy_of_spectrum.npy") ); 
feature_map_avg                      = [];

%% calibration

% camera (in and extrinsic)
pts_fid_3d_all                      = data_struct_calib_module.pts_fid_3d_all; 
pts_fid_2d_left_all                 = data_struct_calib_module.pts_fid_2d_left_all; 
cam_intrinsic_para_left             = data_struct_calib_module.cam_intrinsic_para_left; 
worldPose_left                      = data_struct_calib_module.worldPose_left; 
worldVector_left                    = data_struct_calib_module.worldVector_left; 
tform_oct_to_cam_left               = data_struct_calib_module.tform_oct_to_cam_left; 
pts_fid_2d_right_all                = data_struct_calib_module.pts_fid_2d_right_all; 
cam_intrinsic_para_right            = data_struct_calib_module.cam_intrinsic_para_right; 
worldPose_right                     = data_struct_calib_module.worldPose_right; 
worldVector_right                   = data_struct_calib_module.worldVector_right; 
tform_oct_to_cam_right              = data_struct_calib_module.tform_oct_to_cam_right; 

% tumorid
vec_laser_init_gt_tumorid           = data_struct_calib_module.vec_laser_init_gt_tumorid; 
pts_org_init_gt_tumorid             = data_struct_calib_module.pts_org_init_gt_tumorid; 
list_of_vx_calib_tumorid            = data_struct_calib_module.list_of_vx_calib_tumorid; 
list_of_vy_calib_tumorid            = data_struct_calib_module.list_of_vy_calib_tumorid; 
list_of_vz_calib_tumorid            = data_struct_calib_module.list_of_vz_calib_tumorid; 
alpha_x_opt_tumorid                 = data_struct_calib_module.alpha_x_opt_tumorid;
alpha_y_opt_tumorid                 = data_struct_calib_module.alpha_y_opt_tumorid;
theta_x_opt_tumorid                 = data_struct_calib_module.theta_x_opt_tumorid;
theta_y_opt_tumorid                 = data_struct_calib_module.theta_y_opt_tumorid;
list_of_beta_x_tumorid              = data_struct_calib_module.list_of_beta_x_tumorid;
list_of_beta_y_tumorid              = data_struct_calib_module.list_of_beta_y_tumorid;
vec_laser_opt_tumorid               = data_struct_calib_module.vec_laser_opt_tumorid;
vec_laser_opt_tumorid_vis           = data_struct_calib_module.vec_laser_opt_tumorid_vis; 

% fiber
vec_laser_init_gt_fiber             = data_struct_calib_module.vec_laser_init_gt_fiber;
pts_org_init_gt_fiber               = data_struct_calib_module.pts_org_init_gt_fiber; 
list_of_vx_calib_fiber              = data_struct_calib_module.list_of_vx_calib_fiber; 
list_of_vy_calib_fiber              = data_struct_calib_module.list_of_vy_calib_fiber;
list_of_vz_calib_fiber              = data_struct_calib_module.list_of_vz_calib_fiber;
alpha_x_opt_fiber                   = data_struct_calib_module.alpha_x_opt_fiber;
alpha_y_opt_fiber                   = data_struct_calib_module.alpha_y_opt_fiber;
theta_x_opt_fiber                   = data_struct_calib_module.theta_x_opt_fiber;
theta_y_opt_fiber                   = data_struct_calib_module.theta_y_opt_fiber;
vec_laser_opt_fiber                 = data_struct_calib_module.vec_laser_opt_fiber;
vec_laser_opt_fiber_vis             = data_struct_calib_module.vec_laser_opt_fiber_vis;

%% 3D mapping module 

% oct obj
num_of_knn  = 6; 
scale_dis   = 0.50;             
gridStep    = 0.25;           

path_oct_3d_main                = path_tissue_information;
pts_xyz                         = readNPY( strcat(path_oct_3d_main, "npy_xyz.npy") ); 
pts_rgb                         = readNPY( strcat(path_oct_3d_main, "npy_rgb.npy") ); 
pc_data                         = pointCloud(pts_xyz); 
pc_data.Color                   = uint8(pts_rgb);
pc_data_downsample_raw          = pcdownsample( pc_data, 'gridAverage', gridStep );
pc_data_downsample              = pcdenoise( pc_data_downsample_raw );
flag_figure                     = 1; 
[ cell_pc ]                     = connected_point_cloud_submit( pc_data_downsample.Location, num_of_knn, scale_dis, flag_figure); 

list_of_seg_num                 = [];
for idx_num_pc = 1 : length(cell_pc)
    % return the point cloud object
    list_of_seg_num(idx_num_pc) = length(cell_pc{idx_num_pc});
end
[val_cell, idx_cell]            = sort( list_of_seg_num, "descend" );
[val_max_val_cell, idx_max_val_cell] = max( val_cell ); 
idx_connected_pc                = idx_cell(idx_max_val_cell); 
pts_xyz_redefine                = cell_pc{ idx_connected_pc };
pts_base                        = pointCloud(pts_xyz_redefine).Location; 
pts_obj_use = pts_base; 

%% fig-1: pre-oct  

x_vis_scale = +6.5; 
y_vis_scale = +6.5;
z_vis_scale = +3.5;

[ cell_pc, cell_pc_color ] = connected_point_cloud_with_color_submit( pc_data_downsample.Location, pc_data_downsample.Color, num_of_knn, scale_dis, flag_figure ); 
list_of_seg_num = [];
for idx_num_pc = 1 : length(cell_pc)
    % return the point cloud object
    list_of_seg_num(idx_num_pc) = length(cell_pc{idx_num_pc});
end
[val_cell, idx_cell] = sort( list_of_seg_num, "descend" );
idx_connected_pc = 1; 
pts_xyz_redefine = cell_pc{ idx_cell( idx_connected_pc ) };
pts_rgb_redefine = cell_pc_color{ idx_cell( idx_connected_pc ) };
pts_base = pointCloud(pts_xyz_redefine).Location; 
pts_obj_xyz = pts_base; 
pts_obj_rgb = pts_rgb_redefine; 
pc_data_pre = pc_data;

size_pc = 1000; 
size_axis = 10; 
size_axis_letter = 65; 

%% step-1: ML module

% ML models: chicken-red 
% confidence_scale = 0.10;

confidence_scale = load( strcat(path_tissue_information, "confidence_scale.mat") ).confidence_scale; 
cr_avg_tumor = load( strcat(path_tissue_information, "cr_avg_tumor.mat") ).cr_avg_tumor; 
cr_avg_healthy = load( strcat(path_tissue_information, "cr_avg_healthy.mat") ).cr_avg_healthy; 
cr_avg_thres = load( strcat(path_tissue_information, "cr_avg_thres.mat") ).cr_avg_thres; 

cr_range_min_and_max = abs(cr_avg_healthy - cr_avg_tumor); 
cr_upper_bound_thres = cr_avg_thres + cr_range_min_and_max * confidence_scale; 
cr_lower_bound_thres = cr_avg_thres - cr_range_min_and_max * confidence_scale; 

% classifier from the threshold values
idx_fig_tmp = 3;
label_id_list = []; 
for idx_id = 1 : size(spectral_npy, 1)
    
    % show the spectral signal in 1D graph 
    wavelength_npy_tmp = wavelength_npy(idx_id,:);
    spectral_npy_tmp = spectral_npy(idx_id,:);
    feature_tmp = mean(spectral_npy_tmp); 
    
    % single feature map 
    if feature_tmp >= cr_lower_bound_thres
        
        % tumor
        label_id_tmp = 0;
        
    elseif feature_tmp < cr_upper_bound_thres
        
        % healthy
        label_id_tmp = 1;
       
    end
    label_id_list = [label_id_list; label_id_tmp];

end 

% solution-2
data_tumorid_inference_label = label_id_list;

%% laser spot estimation 

% id laser spots
pts_id_for_match_left_cam = label_center_spot_left;
pts_id_for_match_right_cam = label_center_spot_right;

% project the oct point cloud to the {left} and {right} 
pts_oct_project_to_left = worldToImage( cam_intrinsic_para_left.Intrinsics, tform_oct_to_cam_left.Rotation, tform_oct_to_cam_left.Translation, pts_obj_use );
pts_oct_project_to_right = worldToImage( cam_intrinsic_para_right.Intrinsics, tform_oct_to_cam_right.Rotation, tform_oct_to_cam_right.Translation, pts_obj_use );

% search for the knn-searching index 
[idx_oct_to_cam_left, ~] = knnsearch( pts_oct_project_to_left, pts_id_for_match_left_cam );
[idx_oct_to_cam_right, ~] = knnsearch( pts_oct_project_to_right, pts_id_for_match_right_cam );

% color distribution 
% left color
idx_img_list = 1 : ( size(img_base_left, 1) * size(img_base_left, 2) ); 
[col_img, row_img] = ind2sub( [size(img_base_left, 1), size(img_base_left,2)], idx_img_list ); 
pts_img_idx = [row_img', col_img']; 
[idx_oct_to_color_left, val_oct_to_color_left] = knnsearch( pts_img_idx, pts_oct_project_to_left );
rgb_color_left = reshape(img_base_left, [( size(img_base_left, 1) * size(img_base_left, 2) ), 3]); 
color_to_oct_left = rgb_color_left(idx_oct_to_color_left,:); 

% map the left-color image to the right one
[row_oct_to_color_left, col_oct_to_color_left] = ind2sub( [size(img_base_left, 1), size(img_base_left,2)],  idx_oct_to_color_left );

% laser spot detection
pts_laser_spot_left = load( path_img_spot_centroid_left ).list_of_laser_spot_center;
idx_pts_laser_spot_left_outlier = (pts_laser_spot_left(:,1) == 0) & (pts_laser_spot_left(:,2) == 0); 
pts_laser_spot_left_use = pts_laser_spot_left(~idx_pts_laser_spot_left_outlier, :); 
list_of_pts_laser_spot_tmp_knn = []; 
list_of_pts_laser_spot_tmp_knn_3d = []; 

for idx_laser_spot_track = 1 : size( pts_laser_spot_left_use, 1 ) 

    pts_laser_spot_tmp = double( pts_laser_spot_left_use(idx_laser_spot_track, :) );  
    [idx_laser_spot_knn, val_check] = knnsearch( pts_oct_project_to_left, pts_laser_spot_tmp );
    pts_laser_spot_tmp_knn = pts_oct_project_to_left(idx_laser_spot_knn, :); 

    list_of_pts_laser_spot_tmp_knn = [list_of_pts_laser_spot_tmp_knn; pts_laser_spot_tmp_knn]; 
    list_of_pts_laser_spot_tmp_knn_3d = [list_of_pts_laser_spot_tmp_knn_3d; pts_obj_use(idx_laser_spot_knn, :)];

end 


%% 

% right color 
idx_img_list = 1 : ( size(img_base_right, 1) * size(img_base_right, 2) ); 
[col_img, row_img] = ind2sub( [size(img_base_right, 1), size(img_base_right,2)], idx_img_list ); 
pts_img_idx = [row_img', col_img']; 
[idx_oct_to_color_right, val_oct_to_color_right] = knnsearch( pts_img_idx, pts_oct_project_to_right );
rgb_color_right = reshape(img_base_right, [( size(img_base_right, 1) * size(img_base_right, 2) ), 3]); 
color_to_oct_right = rgb_color_right(idx_oct_to_color_right,:); 

% post-color 
rgb_color_left_post = reshape(img_base_left_post, [( size(img_base_left_post, 1) * size(img_base_left_post, 2) ), 3]); 
color_to_oct_left_post = rgb_color_left_post(idx_oct_to_color_left,:); 
rgb_color_right_post = reshape(img_base_right_post, [( size(img_base_right_post, 1) * size(img_base_right_post, 2) ), 3]); 
color_to_oct_right_post = rgb_color_right_post(idx_oct_to_color_right,:); 

% forward prediction models 
data_of_beta_in_frame = readNPY( strcat( path_tissue_information, "pts_grid_buffer_data.npy" ) );
x_robot_grid_center_tumorid = -120; % -118; 
y_robot_grid_center_tumorid = -32;
beta_list_laser_use_x = data_of_beta_in_frame(:,1) * 1000 - x_robot_grid_center_tumorid;
beta_list_laser_use_y = data_of_beta_in_frame(:,2) * 1000 - y_robot_grid_center_tumorid; 

% laser incidence configurations
traj_laser_roi_scan = [];
traj_laser_roi_vec = []; 
for idx_pts_in_traj = 1 : length(beta_list_laser_use_x) 
    pts_laser_roi_scan_tmp = pts_org_init_gt_tumorid' + ...
                             (alpha_y_opt_tumorid + -( beta_list_laser_use_x(idx_pts_in_traj)) ) * list_of_vy_calib_tumorid + ...
                             (alpha_x_opt_tumorid +    beta_list_laser_use_y(idx_pts_in_traj)  ) * list_of_vx_calib_tumorid;
    traj_laser_roi_scan = [traj_laser_roi_scan; pts_laser_roi_scan_tmp];         
    traj_laser_roi_vec = [traj_laser_roi_vec; vec_laser_opt_tumorid_vis'];
end 

% state estimation: three sources 
pts_oct_3d_from_left = pts_obj_use(idx_oct_to_cam_left, :);
pts_oct_3d_from_right = pts_obj_use(idx_oct_to_cam_right, :);

% get the 3d tissue objects 
x_tissue = pts_obj_use(:,1);
y_tissue = pts_obj_use(:,2);
z_tissue = pts_obj_use(:,3); 
faces = delaunay(x_tissue, y_tissue);       
vertices = [x_tissue(:) y_tissue(:) z_tissue(:)];                 
vert1 = vertices(faces(:,1),:);
vert2 = vertices(faces(:,2),:);
vert3 = vertices(faces(:,3),:);

% errors
list_of_pts_filter = []; 
for idx_id = 1 : size(pts_oct_3d_from_left, 1)  

    % source-1: laser intersection point 
    orig            = traj_laser_roi_scan(idx_id, :);                
    dir             = vec_laser_opt_tumorid';  
    intersect       = TriangleRayIntersection(orig, dir, vert1, vert2, vert3);
    idx_intersect   = find(intersect == 1);
    if length(idx_intersect) > 1
        pts_intersect = vert1(idx_intersect(1), :);
    else
        pts_intersect = vert1(idx_intersect, :);
    end

    % source-2: left configuration
    spot_cen_left_tmp = label_center_spot_left(idx_id,:); 
    idx_cen_cam_and_oct_left = knnsearch(pts_oct_project_to_left, spot_cen_left_tmp); 
    pts_cen_cam_and_oct_left = pts_obj_use(idx_cen_cam_and_oct_left, :); 

    % source-3: right configuration
    spot_cen_right_tmp = label_center_spot_right(idx_id,:); 
    idx_cen_cam_and_oct_right = knnsearch(pts_oct_project_to_right, spot_cen_right_tmp); 
    pts_cen_cam_and_oct_right = pts_obj_use(idx_cen_cam_and_oct_right, :); 

    % present the average coordinate 
    if length(pts_intersect) == 0
        pts_avg = [0.0, 0.0, 0.0]; 
    else 
        pts_avg = ( pts_intersect + pts_cen_cam_and_oct_left + pts_cen_cam_and_oct_right ) ./ 3;
    end
    
    list_of_pts_filter = [list_of_pts_filter; pts_avg]; 
    
end

idx_left_cam_outlier = [];
idx_left_cam_all = ones([size(label_center_spot_left, 1), 1]); 
for idx_left_cam_check = 1 : length(idx_left_cam_all) 
    if ismember(idx_left_cam_check, idx_left_cam_outlier) 
        idx_left_cam_all(idx_left_cam_check) = 0; 
    end 
end
idx_right_cam_outlier = [];
idx_right_cam_all = ones([size(label_center_spot_right, 1),1]); 
for idx_right_cam_check = 1 : length(idx_right_cam_all) 
    if ismember(idx_right_cam_check, idx_right_cam_outlier) 
        idx_right_cam_all(idx_right_cam_check) = 0; 
    end 
end

% inlier of left and right camera
idx_left_and_right_inlier = ( idx_left_cam_all & idx_right_cam_all ); 

% inlier of average filter 
idx_id_in_roi_avg_filter = ~( list_of_pts_filter(:,1) == 0 & list_of_pts_filter(:,2) == 0 & list_of_pts_filter(:,3) == 0 );

% inlier of all 
idx_id_in_roi = ( idx_left_and_right_inlier & idx_id_in_roi_avg_filter ); 

% inlier of all the laser spots 
pts_id_in_roi = list_of_pts_filter(idx_id_in_roi, :); 
label_id_in_roi = data_tumorid_inference_label(idx_id_in_roi);
pts_id_tumor = pts_id_in_roi(label_id_in_roi == 0, :); 
pts_id_healthy = pts_id_in_roi(label_id_in_roi == 1, :); 

%% 3D cutting map 

% healthy region 
[ idx_roi_k, ~ ] = boundary( pts_id_healthy(:,1), pts_id_healthy(:,2) ); 
pts_roi_healthy_k = pts_id_healthy(idx_roi_k, :); 
[ idx_in, ~ ] = inpolygon( pts_obj_use(:,1), pts_obj_use(:,2), pts_roi_healthy_k(:,1), pts_roi_healthy_k(:,2) ); 
pts_roi_healthy_in = pts_obj_use( idx_in, : ); 

% tuomr region
scale_factor_tumor = 0.3;
[ idx_roi_k, ~ ] = boundary( pts_id_tumor(:,1), pts_id_tumor(:,2), scale_factor_tumor); 
pts_roi_tumor_k = pts_id_tumor(idx_roi_k, :); 
[ idx_in, ~ ] = inpolygon( pts_obj_use(:,1), pts_obj_use(:,2), pts_roi_tumor_k(:,1), pts_roi_tumor_k(:,2) ); 
pts_roi_tumor_in = pts_obj_use( idx_in, : ); 

pts_target_cut = pts_roi_tumor_in; 
pc_targert_cut = pointCloud( pts_target_cut ); 
pc_target_cut_down = pcdownsample( pc_targert_cut, 'gridAverage', 0.5 );

% sort the point cloud
idx_traj_for_order_x = pc_target_cut_down.Location(:, 1);
idx_traj_for_order_y = pc_target_cut_down.Location(:, 2);
[ ~, idx_x_sort ] = sort( idx_traj_for_order_x ); 
pts_traj_sorted = pc_target_cut_down.Location(idx_x_sort, :); 
pts_target_final = pts_traj_sorted;

%% kinematics module 

% ik-dof
num_of_var_tumorid = 2;
num_of_var_diode = 2;

% trajectory as targets 
% 3D labelled map
pts_traj_robot_plane = pts_target_final; 

% summary of the IK solvers 
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
    [pts_fk_tmp, ~, ~, ~, ~] = MirrorModel_3D_test( pts_input_tmp, vec_input_tmp, pts_n_tmp, vec_n_tmp );
    list_of_target_ik_pts_fiber = [list_of_target_ik_pts_fiber; pts_fk_tmp']; 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%% fig-3: tags on tumor map

pts_obj_use_vis = pts_obj_use;
mean_pts_obj_use_vis = mean( pts_obj_use_vis );
pts_obj_use_vis(:,1) = pts_obj_use_vis(:,1) - mean_pts_obj_use_vis(1);
pts_obj_use_vis(:,2) = pts_obj_use_vis(:,2) - mean_pts_obj_use_vis(2);
pts_obj_use_vis(:,3) = pts_obj_use_vis(:,3) - mean_pts_obj_use_vis(3);

pts_roi_tumor_k_vis = pts_roi_tumor_k - mean_pts_obj_use_vis;
list_of_pts_laser_spot_tmp_knn_3d_vis = list_of_pts_laser_spot_tmp_knn_3d - mean_pts_obj_use_vis;
pts_list_predict_fiber_vis = pts_list_predict_fiber - mean_pts_obj_use_vis; 

figure(16);clf;
h_pc = scatter3( pts_obj_use_vis(:,1), pts_obj_use_vis(:,2), pts_obj_use_vis(:,3), [], double(color_to_oct_left) ./ 255 , "filled", "SizeData", 300 );
hold on; 
plot3( pts_roi_tumor_k_vis(:,1), pts_roi_tumor_k_vis(:,2), pts_roi_tumor_k_vis(:,3), "r", "LineWidth", 8); 
hold on; 
pcshow(pts_list_predict_fiber_vis, "r", "MarkerSize", size_pc );
hold on;
pcshow( list_of_pts_laser_spot_tmp_knn_3d_vis, "g", "MarkerSize", size_pc );
axis equal;
rotate3d on;
colormap gray;
alpha = 0.30; 
set(h_pc, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha); 
% 
if is_show_axis == "true"
    hold on;
    xlabel("X-mm");
    ylabel("Y-mm");
    zlabel("Z-mm");
end 
% 
xlim( [-x_vis_scale, x_vis_scale] );
ylim( [-y_vis_scale, y_vis_scale] );
zlim( [-z_vis_scale, z_vis_scale] );
% 
set(gca, "FontSize", size_axis_letter ); 
set(gca, "Color", "w");
set(gca, 'XColor',[0.0 0.0 0.0],...
        'XLimitMethod','tight','XTick',[-x_vis_scale 0 x_vis_scale],'YColor',[0.0 0.0 0.0],...
        'YLimitMethod','tight','YTick',[-y_vis_scale 0 y_vis_scale],'ZColor',[0.0 0.0 0.0],...
        'ZLimitMethod','tight','ZTick',[-z_vis_scale 0 z_vis_scale]);
% 
set(gca, "LineWidth", size_axis);
% 
set(gca,'CameraPosition',...
    [-109.555470873472 -100.604617235152 139.798282124067],'CameraUpVector',...
    [0.537014911626919 0.436560617376292 0.721823948097016],'CameraViewAngle',...
    7.31308448784807);
% 
set(gcf, "Color", "w");
set(gcf, 'WindowState', 'maximized');

%% fig-3b: tumor map with the sparse tags

pts_id_tumor_vis    = pts_id_tumor - mean_pts_obj_use_vis; 
pts_id_healthy_vis  = pts_id_healthy - mean_pts_obj_use_vis; 

figure(18);clf; 
h_pc = scatter3( pts_obj_use_vis(:,1), pts_obj_use_vis(:,2), pts_obj_use_vis(:,3), [], double(color_to_oct_left) ./ 255 , "filled", "SizeData", 300 );
hold on;
pcshow(pts_id_tumor_vis, "r", "MarkerSize", 5000 ); 
hold on; 
pcshow(pts_id_healthy_vis, "b", "MarkerSize", 5000 ); 
alpha = 0.30; 
set(h_pc, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha); 
% 
xlim( [-x_vis_scale, x_vis_scale] );
ylim( [-y_vis_scale, y_vis_scale] );
zlim( [-z_vis_scale, z_vis_scale] );
% 
set(gca, "FontSize", size_axis_letter ); 
set(gca, "Color", "w");
set(gca, 'XColor',[0.0 0.0 0.0],...
        'XLimitMethod','tight','XTick',[-x_vis_scale 0 x_vis_scale],'YColor',[0.0 0.0 0.0],...
        'YLimitMethod','tight','YTick',[-y_vis_scale 0 y_vis_scale],'ZColor',[0.0 0.0 0.0],...
        'ZLimitMethod','tight','ZTick',[-z_vis_scale 0 z_vis_scale]);
% 
set(gca, "LineWidth", size_axis);
% 
set(gca,'CameraPosition',...
    [-109.555470873472 -100.604617235152 139.798282124067],'CameraUpVector',...
    [0.537014911626919 0.436560617376292 0.721823948097016],'CameraViewAngle',...
    7.31308448784807);

set(gcf, "Color", "w");
set(gcf, 'WindowState', 'maximized');

