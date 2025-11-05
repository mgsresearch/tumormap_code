

function [ fi, Jacobian_matrix ] = jac_v2_4dof_laser_to_oct_ik_solver_fiber_vis_only_submit( traj_w )

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

    global cell_traj_tracker_only_for_vis;
    global idx_traj_tracker_only_for_vis;

    % current calibrated robot configuration
    theta_z_opt = 0.0;
    theta_vec_degree = [theta_x_opt_fiber, theta_y_opt_fiber, theta_z_opt];
    mat_from_theta_vec_degree = RfromTheta_submit( theta_vec_degree ); 
    vec_laser_opt = mat_from_theta_vec_degree * vec_laser_init_gt_fiber;
    vec_laser_opt = vec_laser_opt ./ norm( vec_laser_opt );

    % property of the robot trajectory
    % num_of_var_diode = 2;
    % num_of_var_fiber
    % length(traj_w)
    num_of_waypoint = length(traj_w) / num_of_var_fiber;

    % function and Jacobian 
    Jacobian_matrix = []; 
    fi = 0.0;
    pts_list_predict_fiber = []; 

    % LS for the trajectory of each waypoints 
    for idx_traj = 1 : num_of_waypoint

        % dof of each robot config
        idx_count = num_of_var_fiber * idx_traj - (num_of_var_fiber - 1);
        beta_x = traj_w(idx_count); 
        beta_y = traj_w(idx_count + 1); 

        % prepare data
        pi_tmp = pts_org_init_gt_fiber + ( alpha_x_opt_fiber + beta_x ) * list_of_vx_calib_fiber' +  ( alpha_y_opt_fiber + beta_y ) * list_of_vy_calib_fiber';
        vi_tmp = vec_laser_opt; 
        pn_tmp = list_of_pn_ik_fiber(idx_traj, :)'; 
        vn_tmp = list_of_vn_ik_fiber(idx_traj, :)';

        % fk 
        [pts_fk, ~, ~, ~, ~] = MirrorModel_3D_test_submit( pi_tmp, vi_tmp, pn_tmp, vn_tmp );

        % derivatives
        diff_data = mirror_diff_3d_test_submit( pi_tmp, vi_tmp, pn_tmp, vn_tmp );

        % diff_pi_to_beta_x 
        diff_pi_to_beta_x = diff_data.diff_p2p * list_of_vx_calib_fiber'; 
        diff_F_to_pi = 2 * (pts_fk' - list_of_target_ik_pts_fiber(idx_traj,:)); 
        diff_F_to_beta_x = diff_F_to_pi * diff_pi_to_beta_x; 

        % diff_pi_to_beta_y 
        diff_pi_to_beta_y = diff_data.diff_p2p * list_of_vy_calib_fiber'; 
        diff_F_to_pi = 2 * (pts_fk' - list_of_target_ik_pts_fiber(idx_traj,:)); 
        diff_F_to_beta_y = diff_F_to_pi * diff_pi_to_beta_y; 

        % jacobian 
        Jacobian_matrix = [Jacobian_matrix; diff_F_to_beta_x; diff_F_to_beta_y];

        % point offsets 
        diff_predict_to_target = pts_fk - list_of_target_ik_pts_fiber(idx_traj, :)'; 
        % norm( diff_predict_to_target ).^2
        fi = fi + norm( diff_predict_to_target ).^2;
        pts_list_predict_fiber = [pts_list_predict_fiber; pts_fk']; 

        % fi

    end 

    % size( fi )

    cell_traj_tracker_only_for_vis{idx_traj_tracker_only_for_vis} = traj_w;
    idx_traj_tracker_only_for_vis = idx_traj_tracker_only_for_vis + 1;

end
