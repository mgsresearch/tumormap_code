

function [ diff_list ] = diff_v_to_theta_submit( vec_input, theta_input )

    % the vector configuration
    vec_x = vec_input(1); 
    vec_y = vec_input(2);
    vec_z = vec_input(3);

    % theta configuration
    theta_x = theta_input(1); 
    theta_y = theta_input(2);
    theta_z = theta_input(3);

    diff_vec_to_theta_x = [
                                                                                                                                                                                                                                                                                                                                                   0
- vec_x*((pi*sin((pi*theta_x)/180)*sin((pi*theta_z)/180))/180 - (pi*cos((pi*theta_x)/180)*cos((pi*theta_z)/180)*sin((pi*theta_y)/180))/180) - vec_y*((pi*cos((pi*theta_z)/180)*sin((pi*theta_x)/180))/180 + (pi*cos((pi*theta_x)/180)*sin((pi*theta_y)/180)*sin((pi*theta_z)/180))/180) - (vec_z*pi*cos((pi*theta_x)/180)*cos((pi*theta_y)/180))/180
  vec_x*((pi*cos((pi*theta_x)/180)*sin((pi*theta_z)/180))/180 + (pi*cos((pi*theta_z)/180)*sin((pi*theta_x)/180)*sin((pi*theta_y)/180))/180) + vec_y*((pi*cos((pi*theta_x)/180)*cos((pi*theta_z)/180))/180 - (pi*sin((pi*theta_x)/180)*sin((pi*theta_y)/180)*sin((pi*theta_z)/180))/180) - (vec_z*pi*cos((pi*theta_y)/180)*sin((pi*theta_x)/180))/180
    ];

    diff_vec_to_theta_y = [
                                             (vec_z*pi*cos((pi*theta_y)/180))/180 - (vec_x*pi*cos((pi*theta_z)/180)*sin((pi*theta_y)/180))/180 + (vec_y*pi*sin((pi*theta_y)/180)*sin((pi*theta_z)/180))/180
(vec_z*pi*sin((pi*theta_x)/180)*sin((pi*theta_y)/180))/180 + (vec_x*pi*cos((pi*theta_y)/180)*cos((pi*theta_z)/180)*sin((pi*theta_x)/180))/180 - (vec_y*pi*cos((pi*theta_y)/180)*sin((pi*theta_x)/180)*sin((pi*theta_z)/180))/180
(vec_y*pi*cos((pi*theta_x)/180)*cos((pi*theta_y)/180)*sin((pi*theta_z)/180))/180 - (vec_x*pi*cos((pi*theta_x)/180)*cos((pi*theta_y)/180)*cos((pi*theta_z)/180))/180 - (vec_z*pi*cos((pi*theta_x)/180)*sin((pi*theta_y)/180))/180
    ]; 

    diff_vec_to_theta_z = [
                                                                                                                                                            - (vec_y*pi*cos((pi*theta_y)/180)*cos((pi*theta_z)/180))/180 - (vec_x*pi*cos((pi*theta_y)/180)*sin((pi*theta_z)/180))/180
vec_x*((pi*cos((pi*theta_x)/180)*cos((pi*theta_z)/180))/180 - (pi*sin((pi*theta_x)/180)*sin((pi*theta_y)/180)*sin((pi*theta_z)/180))/180) - vec_y*((pi*cos((pi*theta_x)/180)*sin((pi*theta_z)/180))/180 + (pi*cos((pi*theta_z)/180)*sin((pi*theta_x)/180)*sin((pi*theta_y)/180))/180)
vec_x*((pi*cos((pi*theta_z)/180)*sin((pi*theta_x)/180))/180 + (pi*cos((pi*theta_x)/180)*sin((pi*theta_y)/180)*sin((pi*theta_z)/180))/180) - vec_y*((pi*sin((pi*theta_x)/180)*sin((pi*theta_z)/180))/180 - (pi*cos((pi*theta_x)/180)*cos((pi*theta_z)/180)*sin((pi*theta_y)/180))/180)
     
    ];

    diff_list = [];
    diff_list.diff_vec_to_theta_x = diff_vec_to_theta_x;
    diff_list.diff_vec_to_theta_y = diff_vec_to_theta_y;
    diff_list.diff_vec_to_theta_z = diff_vec_to_theta_z;

end 

