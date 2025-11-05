
clc; 
clear all; 
close all;

% load the data file
path_calib_oct_to_axis_file = "./dataset/calibration/calib_laser_axis_use_1.mat"; 
load( path_calib_oct_to_axis_file );

%% calibration

list_of_fid_axis_3d_unflip = data_struct.list_of_fid_axis_3d_unflip;

figure(1);
hold on;
pcshow( list_of_fid_axis_3d_unflip, "r", "MarkerSize", 3000 );
xlim([-20, +20]);
ylim([-20, +20]);
zlim([-10, +10]); 
xlabel("X (unit: mm)"); 
ylabel("Y (unit: mm)");
zlabel("Z (unit: mm)");
set(gca, "FontSize", 20);

% use the manual label steps
len_vec_scale_vis_onlye = 5.0;
vx = [list_of_fid_axis_3d_unflip(7,:) - list_of_fid_axis_3d_unflip(1,:)];
vx = vx ./ norm( vx ); 
vx_vis = vx .* len_vec_scale_vis_onlye; 
vy = [list_of_fid_axis_3d_unflip(3,:) - list_of_fid_axis_3d_unflip(1,:)];
vy = vy ./ norm( vy );
vy_vis = vy .* len_vec_scale_vis_onlye; 
vz = cross(vx, vy); 
vz = vz ./ norm( vz );
vz_vis = vz .* len_vec_scale_vis_onlye; 

% vis + label
figure(1);
hold on;
quiver3( list_of_fid_axis_3d_unflip(1,1), list_of_fid_axis_3d_unflip(1,2), list_of_fid_axis_3d_unflip(1,3), vx_vis(1), vx_vis(2), vx_vis(3), "r", "LineWidth", 3); 
hold on;
quiver3( list_of_fid_axis_3d_unflip(1,1), list_of_fid_axis_3d_unflip(1,2), list_of_fid_axis_3d_unflip(1,3), vy_vis(1), vy_vis(2), vy_vis(3), "g", "LineWidth", 3); 
hold on; 
quiver3( list_of_fid_axis_3d_unflip(1,1), list_of_fid_axis_3d_unflip(1,2), list_of_fid_axis_3d_unflip(1,3), vz_vis(1), vz_vis(2), vz_vis(3), "b", "LineWidth", 3); 
