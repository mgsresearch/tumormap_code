
function [diff_data] = mirror_diff_3d_test_submit( pSi, vSi, pNi, vNi)

    % argvs: 
        % pts_laser_input (3 x 1): input laser origin (center)
        % vec_laser_input (3 x 1): input laser direction (vector)
        % pts_mirror_center (3 x 1): mirror center
        % vec_mirror_normal (3 x 1): mirror direction
        % vec_mirror_rotate (3 x 1): mirror rotation axis
        % vec_mirror_init (3 x 1): mirror initial axis
        % theta_rotate (1 x 1): mirror rotated angle
    % outputs: 
        % diff_data: all the analytical gradients
   

    % define the analytical derivative
    diff_data = [];

    % diff: p to pts-input
    diff_p2p = eye(3) + ( -1 ./ dot( vNi, vSi ) ) * vSi * vNi';
    diff_data.diff_p2p = diff_p2p;

    % diff: p to vec-input
    k1 = - dot( vSi, vNi ) * dot( vNi, pSi - pNi ) ./ ( dot( vSi, vNi ) )^2; 
    k2 = + dot( vNi, pSi - pNi ) ./ ( dot( vSi, vNi ) )^2; 
    diff_p2v = zeros( [3, 3] ) + k1 * eye(3) + k2 * vSi * vNi';
    diff_data.diff_p2v = diff_p2v;

    % diff: p to pts-mirror-center
    diff_p2pN = ( + 1 ./ dot( vNi, vSi ) ) * vSi * vNi';
    diff_data.diff_p2pN = diff_p2pN;

    % diff: p to vec-mirror-rotate-axis
    k1 = - 1 ./ dot( vNi, vSi );
    k2 = + (dot( vNi, pSi - pNi )) ./ ( dot( vNi, vSi ) )^2; 
    diff_p2vN = k1 * vSi * ( pSi - pNi )' + k2 * vSi * vSi';
    % diff_vN2vk = Jac_gradient_symbolic(theta_i, vIi(1), vIi(2), vIi(3), vKi(1), vKi(2), vKi(3));
    % diff_p2vk = diff_p2vN * diff_vN2vk; 
    diff_data.diff_p2vN = diff_p2vN;

    % diff: v to pts-input
    diff_v2p = zeros( [3, 3] );
    diff_data.diff_v2p = diff_v2p;

    % diff: v to vec-input
    M_v2v = diff_V1V2V2_V1_3D(vSi, vNi);
    diff_v2v = eye(3) + (-2) * M_v2v;
    diff_data.diff_v2v = diff_v2v;

    % diff: v to pts-mirror-center
    diff_v2pN = zeros( [3, 3] );
    diff_data.diff_v2pN = diff_v2pN; 

    % diff: v to vec-mirror-normal 
    diff_v2vN = -2 * diff_V1V2V2_V2_3D(vSi, vNi); 
    % diff_v2vk = diff_v2vN * diff_vN2vk;
    diff_data.diff_v2vN = diff_v2vN; 

    % diff: g to pts-input
    diff_g2p = diff_p2p - eye(3); 
    diff_data.diff_g2p = diff_g2p;

    % diff: g to vec-input
    diff_g2v = diff_p2v;
    diff_data.diff_g2v = diff_g2v;

    % diff: g to pts-mirror-center
    diff_g2pN = diff_p2pN;
    diff_data.diff_g2pN = diff_g2pN;

    % diff: g to vec-mirror-normal
    diff_g2vN = diff_p2vN;
    diff_data.diff_g2vN = diff_g2vN;

    % diff: d to pts-input
    % diff: d to vec-input
    % diff: d to pts-mirror-center
    % diff: d to vec-mirror-normal
    scale_g = - dot( pSi - pNi, vNi ) ./ dot( vSi, vNi ) ;
    g = scale_g * vSi; 
    diff_d2g = g / norm( g );
    diff_d2p = diff_d2g' * diff_g2p;
    diff_d2v = diff_d2g' * diff_g2v;
    diff_d2pN = diff_d2g' * diff_g2pN; 
    diff_d2vN = diff_d2g' * diff_g2vN;

    % save the data 
    diff_data.diff_d2p = diff_d2p; 
    diff_data.diff_d2v = diff_d2v;
    diff_data.diff_d2pN = diff_d2pN; 
    diff_data.diff_d2vN = diff_d2vN;

end
