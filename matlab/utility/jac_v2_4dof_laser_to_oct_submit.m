

function [ list_fi, Jacobian_matrix ] = jac_v2_4dof_laser_to_oct_submit( w_var )

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

    % set the parameter variables 
    alpha_x         = w_var(1); 
    alpha_y         = w_var(2);
    theta_x_tmp     = w_var(3); 
    theta_y_tmp     = w_var(4); 
    theta_z_tmp     = 0.0;

    % get the current laser configuration
    theta_vec_degree            = [theta_x_tmp, theta_y_tmp, theta_z_tmp];
    mat_from_theta_vec_degree   = RfromTheta_submit( theta_vec_degree ); 
    vec_laser_tmp               = mat_from_theta_vec_degree * vec_laser_init_gt;
    vec_laser_tmp               = vec_laser_tmp ./ norm( vec_laser_tmp );

    % function and Jacobian 
    Jacobian_matrix     = []; 
    list_fi             = [];
    pts_list_predict    = []; 
    
    for idx_tmp = 1 : size( list_of_target_pts, 1 )
    
        % prepare data
        pi_tmp = pts_org_init_gt + ( alpha_x + list_of_beta_x(idx_tmp) ) * list_of_vx_calib' + ...
                                   ( alpha_y + list_of_beta_y(idx_tmp) ) * list_of_vy_calib';
        vi_tmp = vec_laser_tmp; 
        pn_tmp = list_of_pn(idx_tmp, :)'; 
        vn_tmp = list_of_vn(idx_tmp, :)';
        
        % fk 
        [pts_fk, ~, ~, ~, ~] = MirrorModel_3D_test_submit( pi_tmp, vi_tmp, pn_tmp, vn_tmp );

        % derivatives
        diff_data = mirror_diff_3d_test_submit( pi_tmp, vi_tmp, pn_tmp, vn_tmp );

        % diff_pi_to_alpha_x 
        diff_pi_to_alpha_x = diff_data.diff_p2p * list_of_vx_calib'; 

        % diff_pi_to_alpha_y 
        diff_pi_to_alpha_y = diff_data.diff_p2p * list_of_vy_calib'; 

        % diff_v_to_theta 
        diff_list = diff_v_to_theta_submit( vec_laser_init_gt, theta_vec_degree );

        % diff_pi_to_theta_x 
        diff_v_to_theta_x = diff_list.diff_vec_to_theta_x;
        diff_pi_to_theta_x = diff_data.diff_p2v * diff_v_to_theta_x; 

        % diff_pi_to_theta_y
        diff_v_to_theta_y = diff_list.diff_vec_to_theta_y; 
        diff_pi_to_theta_y = diff_data.diff_p2v * diff_v_to_theta_y; 

        % jacobian 
        Jacobian_matrix = [Jacobian_matrix; diff_pi_to_alpha_x, diff_pi_to_alpha_y, diff_pi_to_theta_x, diff_pi_to_theta_y];

        % point offsets 
        fi = pts_fk - list_of_target_pts(idx_tmp, :)';
        pts_list_predict = [pts_list_predict; pts_fk']; 
        list_fi = [list_fi; fi];
 
    end 

end
