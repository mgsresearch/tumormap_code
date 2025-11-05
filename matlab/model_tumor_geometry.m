
clc; 
clear all; 
close all; 

%% calibration module 

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

% laser-object definition
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

%% start the mice all-in-one workflow 

% initialize definitions
idx_of_laser                        = "fiber"; 
num_of_var_fiber                    = 2;
path_tissue_information             = "./dataset/mice/sts_tumor_example/"; 
path_global_calib                   = "./dataset/calibration/"; 

% system calibration  
date_of_file                = "use_4";  
path_save_data_base         = path_global_calib;
path_calib_module_file      = strcat(path_save_data_base, "data_struct_calib_module_", date_of_file, ".mat"); 
data_struct_calib_module    = load( path_calib_module_file ).data_struct_calib_module; 

%% load the calibration information 

% camera (in and extrinsic)
pts_fid_3d_all              = data_struct_calib_module.pts_fid_3d_all; 
pts_fid_2d_left_all         = data_struct_calib_module.pts_fid_2d_left_all; 
cam_intrinsic_para_left     = data_struct_calib_module.cam_intrinsic_para_left; 
worldPose_left              = data_struct_calib_module.worldPose_left; 
worldVector_left            = data_struct_calib_module.worldVector_left; 
tform_oct_to_cam_left       = data_struct_calib_module.tform_oct_to_cam_left; 
pts_fid_2d_right_all        = data_struct_calib_module.pts_fid_2d_right_all; 
cam_intrinsic_para_right    = data_struct_calib_module.cam_intrinsic_para_right; 
worldPose_right             = data_struct_calib_module.worldPose_right; 
worldVector_right           = data_struct_calib_module.worldVector_right; 
tform_oct_to_cam_right      = data_struct_calib_module.tform_oct_to_cam_right; 

% tumorid
vec_laser_init_gt_tumorid   = data_struct_calib_module.vec_laser_init_gt_tumorid; 
pts_org_init_gt_tumorid     = data_struct_calib_module.pts_org_init_gt_tumorid; 
list_of_vx_calib_tumorid    = data_struct_calib_module.list_of_vx_calib_tumorid; 
list_of_vy_calib_tumorid    = data_struct_calib_module.list_of_vy_calib_tumorid; 
list_of_vz_calib_tumorid    = data_struct_calib_module.list_of_vz_calib_tumorid; 
alpha_x_opt_tumorid         = data_struct_calib_module.alpha_x_opt_tumorid;
alpha_y_opt_tumorid         = data_struct_calib_module.alpha_y_opt_tumorid;
theta_x_opt_tumorid         = data_struct_calib_module.theta_x_opt_tumorid;
theta_y_opt_tumorid         = data_struct_calib_module.theta_y_opt_tumorid;
list_of_beta_x_tumorid      = data_struct_calib_module.list_of_beta_x_tumorid;
list_of_beta_y_tumorid      = data_struct_calib_module.list_of_beta_y_tumorid;
vec_laser_opt_tumorid       = data_struct_calib_module.vec_laser_opt_tumorid;
vec_laser_opt_tumorid_vis   = data_struct_calib_module.vec_laser_opt_tumorid_vis; 

% fiber
vec_laser_init_gt_fiber     = data_struct_calib_module.vec_laser_init_gt_fiber;
pts_org_init_gt_fiber       = data_struct_calib_module.pts_org_init_gt_fiber; 
list_of_vx_calib_fiber      = data_struct_calib_module.list_of_vx_calib_fiber; 
list_of_vy_calib_fiber      = data_struct_calib_module.list_of_vy_calib_fiber;
list_of_vz_calib_fiber      = data_struct_calib_module.list_of_vz_calib_fiber;
alpha_x_opt_fiber           = data_struct_calib_module.alpha_x_opt_fiber;
alpha_y_opt_fiber           = data_struct_calib_module.alpha_y_opt_fiber;
theta_x_opt_fiber           = data_struct_calib_module.theta_x_opt_fiber;
theta_y_opt_fiber           = data_struct_calib_module.theta_y_opt_fiber;
vec_laser_opt_fiber         = data_struct_calib_module.vec_laser_opt_fiber;
vec_laser_opt_fiber_vis     = data_struct_calib_module.vec_laser_opt_fiber_vis;

% Tumorid: 2d laser spots + reference images
label_center_spot_left      = readNPY( strcat(path_tissue_information, "label_center_spot_left.npy") ); 
label_center_spot_right     = readNPY( strcat(path_tissue_information, "label_center_spot_right.npy") ); 

img_base_left               = load( strcat(path_tissue_information, "img_base_left.mat") ).img_base_left;
img_base_right              = load( strcat(path_tissue_information, "img_base_right.mat") ).img_base_right;

% post-scanning 
img_base_left_post          = load( strcat(path_tissue_information, "img_base_left_post.mat") ).img_base_left_post;
img_base_right_post         = load( strcat(path_tissue_information, "img_base_right_post.mat") ).img_base_right_post;

% return;

%% 3D mapping module 

% oct obj
num_of_knn                  = 6; 
scale_dis                   = 0.50;  % unit: mm
gridStep                    = 0.25;  % unit: mm 
path_oct_3d_main            = path_tissue_information;
pts_xyz                     = readNPY( strcat(path_oct_3d_main, "npy_xyz.npy") ); 
pts_rgb                     = readNPY( strcat(path_oct_3d_main, "npy_rgb.npy") ); 
pc_data                     = pointCloud(pts_xyz); 
pc_data.Color               = uint8(pts_rgb);
pc_data_downsample_raw      = pcdownsample( pc_data, 'gridAverage', gridStep );
pc_data_downsample          = pcdenoise( pc_data_downsample_raw );
flag_figure                 = 1; 
[ cell_pc ]                 = connected_point_cloud_submit( pc_data_downsample.Location, num_of_knn, scale_dis, flag_figure); 

% optimal point cloud object selection
% optimal index
% rank/sort the point cloud index. 
list_of_seg_num = [];
for idx_num_pc = 1 : length(cell_pc)
    list_of_seg_num(idx_num_pc) = length(cell_pc{idx_num_pc});
end
[val_cell, idx_cell] = sort( list_of_seg_num, "descend" );
idx_connected_pc = 1; 
pts_xyz_redefine = cell_pc{ idx_cell( idx_connected_pc ) };
pts_base = pointCloud(pts_xyz_redefine).Location; 
pts_obj_use = pts_base; 

%% tumorid predictions 
% python
% MLP fronm CPU-pytorch 
path_tumorid_inference_test = strcat(path_tissue_information, "mlp_inference_testdata.npy");
data_tumorid_inference_label = readNPY(path_tumorid_inference_test); 

%% state estimation 

% id laser spots
pts_id_for_match_left_cam = label_center_spot_left;
pts_id_for_match_right_cam = label_center_spot_right;

% project the oct point cloud to the {left} and {right} 
pts_oct_project_to_left = worldToImage( cam_intrinsic_para_left.Intrinsics, tform_oct_to_cam_left.Rotation, tform_oct_to_cam_left.Translation, pts_obj_use );
pts_oct_project_to_right = worldToImage( cam_intrinsic_para_right.Intrinsics, tform_oct_to_cam_right.Rotation, tform_oct_to_cam_right.Translation, pts_obj_use );

% search for the knn-searching index 
[idx_oct_to_cam_left, ~] = knnsearch( pts_oct_project_to_left, pts_id_for_match_left_cam );
[idx_oct_to_cam_right, ~] = knnsearch( pts_oct_project_to_right, pts_id_for_match_right_cam );

% forward prediction models 
data_of_beta_in_frame = readNPY( strcat( path_tissue_information, "pts_grid_buffer_data.npy" ) );

x_robot_grid_center_tumorid = -118; 
y_robot_grid_center_tumorid = -32;
beta_list_laser_use_x = data_of_beta_in_frame(:,1) * 1000 - x_robot_grid_center_tumorid;
beta_list_laser_use_y = data_of_beta_in_frame(:,2) * 1000 - y_robot_grid_center_tumorid; 
traj_laser_roi_scan = [];
traj_laser_roi_vec = []; 
for idx_pts_in_traj = 1 : length(beta_list_laser_use_x) 
    pts_laser_roi_scan_tmp = pts_org_init_gt_tumorid' + ...
                             (alpha_y_opt_tumorid + -( beta_list_laser_use_x(idx_pts_in_traj)) ) * list_of_vy_calib_tumorid + ...
                             (alpha_x_opt_tumorid +    beta_list_laser_use_y(idx_pts_in_traj)  ) * list_of_vx_calib_tumorid;
    traj_laser_roi_scan = [traj_laser_roi_scan; pts_laser_roi_scan_tmp];         
    traj_laser_roi_vec = [traj_laser_roi_vec; vec_laser_opt_tumorid_vis'];
end 

% left color
idx_img_list = 1 : ( size(img_base_left, 1) * size(img_base_left, 2) ); 
[col_img, row_img] = ind2sub( [size(img_base_left, 1), size(img_base_left,2)], idx_img_list ); 
pts_img_idx = [row_img', col_img']; 
[idx_oct_to_color_left, val_oct_to_color_left] = knnsearch( pts_img_idx, pts_oct_project_to_left );
rgb_color_left = reshape(img_base_left, [( size(img_base_left, 1) * size(img_base_left, 2) ), 3]); 
color_to_oct_left = rgb_color_left(idx_oct_to_color_left,:); 

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

%% revisit-1: forward kinematics 

% source-1: laser intersection point 
idx_id          = 28; 
orig            = traj_laser_roi_scan(idx_id, :);        
dir             = vec_laser_opt_tumorid';  
vec_dir_vis     = dir * 8;
intersect       = TriangleRayIntersection(orig, dir, vert1, vert2, vert3);
idx_intersect   = find(intersect == 1);
if length(idx_intersect) > 1
    pts_intersect = vert1(idx_intersect(1), :);
else
    pts_intersect = vert1(idx_intersect, :);
end

scale_len_vector = 3;
list_of_vx_calib_fiber_vis = list_of_vx_calib_fiber * scale_len_vector; 
list_of_vy_calib_fiber_vis = list_of_vy_calib_fiber * scale_len_vector; 
list_of_vz_calib_fiber_vis = list_of_vz_calib_fiber * scale_len_vector; 

tform_oct_in_world                = eye(4); 

tform_cam_left_in_world           = eye(4);
tform_cam_left_in_world(1:3,1:3)  = tform_oct_to_cam_left.Rotation;
tform_cam_left_in_world(1:3,4)    = tform_oct_to_cam_left.Translation;
config_cam_left_in_world          = vis_3d_frame( tform_cam_left_in_world ); 
pts_cam_left_in_world             = config_cam_left_in_world.pts_org_input_in_world(1:3) ; 
vec_cam_left_in_world_x           = config_cam_left_in_world.vec_x_input_in_world(1:3) * scale_len_vector; 
vec_cam_left_in_world_y           = config_cam_left_in_world.vec_y_input_in_world(1:3) * scale_len_vector;
vec_cam_left_in_world_z           = config_cam_left_in_world.vec_z_input_in_world(1:3) * scale_len_vector; 

tform_cam_right_in_world          = eye(4);
tform_cam_right_in_world(1:3,1:3) = tform_oct_to_cam_right.Rotation;
tform_cam_right_in_world(1:3,4)   = tform_oct_to_cam_right.Translation;
config_cam_right_in_world         = vis_3d_frame( tform_cam_right_in_world ); 
pts_cam_right_in_world            = config_cam_right_in_world.pts_org_input_in_world(1:3) ; 
vec_cam_right_in_world_x          = config_cam_right_in_world.vec_x_input_in_world(1:3) * scale_len_vector; 
vec_cam_right_in_world_y          = config_cam_right_in_world.vec_y_input_in_world(1:3) * scale_len_vector;
vec_cam_right_in_world_z          = config_cam_right_in_world.vec_z_input_in_world(1:3) * scale_len_vector; 

figure(28);clf; 

% 
pcshow(pts_obj_use, "MarkerSize", 800); 
hold on; 
pcshow( pts_intersect, "r", "MarkerSize", 2000 );
hold on;
pcshow( orig, "r", "MarkerSize", 2000 );
hold on;
quiver3( orig(1), orig(2), orig(3), vec_dir_vis(1), vec_dir_vis(2), vec_dir_vis(3), "r", "filled", "LineWidth", 3);

% laser configuration 
hold on;
pcshow( pts_org_init_gt_fiber', "r", "MarkerSize", 2000 );
hold on;
quiver3( pts_org_init_gt_fiber(1), pts_org_init_gt_fiber(2), pts_org_init_gt_fiber(3), list_of_vx_calib_fiber_vis(1), list_of_vx_calib_fiber_vis(2), list_of_vx_calib_fiber_vis(3), "r", "filled", "LineWidth", 3);
hold on;
quiver3( pts_org_init_gt_fiber(1), pts_org_init_gt_fiber(2), pts_org_init_gt_fiber(3), list_of_vy_calib_fiber_vis(1), list_of_vy_calib_fiber_vis(2), list_of_vy_calib_fiber_vis(3), "g", "filled", "LineWidth", 3);
hold on;
quiver3( pts_org_init_gt_fiber(1), pts_org_init_gt_fiber(2), pts_org_init_gt_fiber(3), list_of_vz_calib_fiber_vis(1), list_of_vz_calib_fiber_vis(2), list_of_vz_calib_fiber_vis(3), "b", "filled", "LineWidth", 3);

% % left camera
% hold on;
% pcshow( pts_cam_left_in_world, "r", "MarkerSize", 2000 );
% hold on;
% quiver3( pts_cam_left_in_world(1), pts_cam_left_in_world(2), pts_cam_left_in_world(3), vec_cam_left_in_world_x(1), vec_cam_left_in_world_x(2), vec_cam_left_in_world_x(3), "r", "filled", "LineWidth", 3);
% hold on;
% quiver3( pts_cam_left_in_world(1), pts_cam_left_in_world(2), pts_cam_left_in_world(3), vec_cam_left_in_world_y(1), vec_cam_left_in_world_y(2), vec_cam_left_in_world_y(3), "g", "filled", "LineWidth", 3);
% hold on;
% quiver3( pts_cam_left_in_world(1), pts_cam_left_in_world(2), pts_cam_left_in_world(3), vec_cam_left_in_world_z(1), vec_cam_left_in_world_z(2), vec_cam_left_in_world_z(3), "b", "filled", "LineWidth", 3);
% 
% % right camera
% hold on;
% pcshow( pts_cam_right_in_world, "r", "MarkerSize", 2000 );
% hold on;
% quiver3( pts_cam_right_in_world(1), pts_cam_right_in_world(2), pts_cam_right_in_world(3), vec_cam_right_in_world_x(1), vec_cam_right_in_world_x(2), vec_cam_right_in_world_x(3), "r", "filled", "LineWidth", 3);
% hold on;
% quiver3( pts_cam_right_in_world(1), pts_cam_right_in_world(2), pts_cam_right_in_world(3), vec_cam_right_in_world_y(1), vec_cam_right_in_world_y(2), vec_cam_right_in_world_y(3), "g", "filled", "LineWidth", 3);
% hold on;
% quiver3( pts_cam_right_in_world(1), pts_cam_right_in_world(2), pts_cam_right_in_world(3), vec_cam_right_in_world_z(1), vec_cam_right_in_world_z(2), vec_cam_right_in_world_z(3), "b", "filled", "LineWidth", 3);

colormap("gray"); 
xlabel("X (unit: mm)"); 
ylabel("Y (unit: mm)");
zlabel("Z (unit: mm)");
title("laser position estimation ");
set(gca, "FontSize", 20); 

%% tumor tags

figure(5);clf; 
h_vis_main = scatter3( pts_xyz_redefine(:,1), pts_xyz_redefine(:,2), pts_xyz_redefine(:,3), [], double( color_to_oct_left ) ./ 255 , "filled", "SizeData", 300 );
rotate3d on;
alpha = 0.25;
set( h_vis_main, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha); 
title("tumor tags")

% end-to-end error reports
% report the end-to-end errors
list_of_pts_filter = []; 
for idx_id = 1 : size(pts_oct_3d_from_left, 1)  

    % source-1: laser intersection point 
    orig  = traj_laser_roi_scan(idx_id, :);                
    dir   = vec_laser_opt_tumorid';  
    intersect = TriangleRayIntersection(orig, dir, vert1, vert2, vert3);
    idx_intersect = find(intersect == 1);
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

    if pts_avg == [0.0, 0.0, 0.0]
        continue; 
    end 

    % vis of the model
    figure(5);
    hold on; 
    if (data_tumorid_inference_label(idx_id) == 1) 
        % healthy = 1 -> "blue"
        pcshow(pts_avg, "b", "MarkerSize", 3500); 
    end
    if (data_tumorid_inference_label(idx_id) == 0)
        % tumor = 0 -> "red" 
        pcshow(pts_avg, "r", "MarkerSize", 3500); 
    end
    
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

idx_left_and_right_inlier = ( idx_left_cam_all & idx_right_cam_all ); 

idx_id_in_roi = ~( list_of_pts_filter(:,1) == 0 & list_of_pts_filter(:,2) == 0 & list_of_pts_filter(:,3) == 0 );

idx_id_in_roi = ( idx_left_and_right_inlier & idx_id_in_roi ); 
 
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
[ idx_roi_k, ~ ] = boundary( pts_id_tumor(:,1), pts_id_tumor(:,2) ); 
pts_roi_tumor_k = pts_id_tumor(idx_roi_k, :); 
[ idx_in, ~ ] = inpolygon( pts_obj_use(:,1), pts_obj_use(:,2), pts_roi_tumor_k(:,1), pts_roi_tumor_k(:,2) ); 
pts_roi_tumor_in = pts_obj_use( idx_in, : ); 

%% revisit: tumor geoetry

figure(21);clf; 
scatter( pts_obj_use(:,1), pts_obj_use(:,2), [], [0.5 0.5 0.5], 'filled', "SizeData", 100 ); 
hold on; 
scatter( pts_id_tumor(:,1), pts_id_tumor(:,2), [], "r", "LineWidth", 12); 
hold on; 
scatter( pts_id_healthy(:,1), pts_id_healthy(:,2), [], "b", "LineWidth", 12); 
grid on;
axis equal; 
alpha 0.5;

figure(23);clf; 
scatter( pts_obj_use(:,1), pts_obj_use(:,2), [], [0.5 0.5 0.5], 'filled', "SizeData", 100 ); 
hold on; 
scatter( pts_id_tumor(:,1), pts_id_tumor(:,2), [], "r", "LineWidth", 12); 
hold on; 
scatter( pts_id_healthy(:,1), pts_id_healthy(:,2), [], "b", "LineWidth", 12); 
hold on;
plot( pts_roi_tumor_k(:,1), pts_roi_tumor_k(:,2), "r", "LineWidth", 6 ); 
% hold on;
% scatter( pts_roi_tumor_in(:,1), pts_roi_tumor_in(:,2), [], "g" ); 
grid on;
axis equal; 
alpha 0.5;

figure(24);clf; 
scatter( pts_obj_use(:,1), pts_obj_use(:,2), [], [0.5 0.5 0.5], 'filled', "SizeData", 100 ); 
hold on; 
% scatter( pts_id_tumor(:,1), pts_id_tumor(:,2), [], "r", "LineWidth", 6); 
% hold on; 
% scatter( pts_id_healthy(:,1), pts_id_healthy(:,2), [], "b", "LineWidth", 6); 
hold on;
plot( pts_roi_tumor_k(:,1), pts_roi_tumor_k(:,2), "r", "LineWidth", 12 ); 
hold on;
scatter( pts_roi_tumor_in(:,1), pts_roi_tumor_in(:,2), [], "g" ); 
grid on;
axis equal; 
alpha 0.5;

%% 

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

global cell_traj_tracker_only_for_vis;
global idx_traj_tracker_only_for_vis;
idx_traj_tracker_only_for_vis = 1;

% ik-dof
num_of_var_tumorid = 2;
num_of_var_diode = 2;

% trajectory as targets 
% 3D labelled map
pts_traj_robot_plane = pts_target_final; 

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
    [pts_fk_tmp, ~, ~, ~, ~] = MirrorModel_3D_test( pts_input_tmp, vec_input_tmp, pts_n_tmp, vec_n_tmp );
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

% fiber: registration to robot
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

%% tumor boundary 

pts_xyz_for_fig_system       = pts_xyz_redefine - mean( pts_xyz_redefine, 1 );
pts_rgb_for_fig_system       = color_to_oct_left;
pts_list_predict_fiber_vis   = pts_list_predict_fiber - mean( pts_xyz_redefine, 1 );
pts_roi_tumor_k_vis          = pts_roi_tumor_k - mean( pts_xyz_redefine, 1 ); 

figure(28);clf; 
h_vis_main = scatter3( pts_xyz_for_fig_system(:,1), pts_xyz_for_fig_system(:,2), pts_xyz_for_fig_system(:,3), [], double( color_to_oct_left_post ) ./ 255 , "filled", "SizeData", 300 );
hold on; 
pcshow( pts_roi_tumor_k_vis, "r", "MarkerSize", 1500);
hold on; 
plot3( pts_roi_tumor_k_vis(:,1), pts_roi_tumor_k_vis(:,2), pts_roi_tumor_k_vis(:,3), "c-", "LineWidth", 15); 
hold on; 
rotate3d on;
alpha = 0.20;
set( h_vis_main, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha); 
% viewpoint 
    set(gca,'CameraUpVector',[0 0 1],'DataAspectRatio',[1 1 1],'Projection',...
    'perspective','XColor',[1 1 1],'YColor',[1 1 1],...
    'ZColor',[1 1 1]);
    set(gca,"Color", "w");
    set(gcf,"Color", "w");

figure(26);clf; 
h_vis_main = scatter3( pts_xyz_for_fig_system(:,1), pts_xyz_for_fig_system(:,2), pts_xyz_for_fig_system(:,3), [], double( pts_rgb_for_fig_system ) ./ 255 , "filled", "SizeData", 300 );
hold on; 
% plot3( pts_list_predict_fiber_vis(:,1), pts_list_predict_fiber_vis(:,2), pts_list_predict_fiber_vis(:,3), "r-" );
hold on; 
pcshow( pts_list_predict_fiber_vis, "r", "MarkerSize", 1000);
% hold on; 
% plot3( pts_roi_tumor_k_vis(:,1), pts_roi_tumor_k_vis(:,2), pts_roi_tumor_k_vis(:,3), "y-", "LineWidth", 10); 
hold on; 
rotate3d on;
alpha = 0.25;
set( h_vis_main, 'MarkerEdgeAlpha', alpha, 'MarkerFaceAlpha', alpha); 