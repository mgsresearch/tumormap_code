
function [data_plane] = get_Plane_submit(vNi, pNi, scale_plane) 

    % controlled vectors
    % Two parallel vector in the plane
    % Vectors on the surfaces
    x_val = -( vNi(1) * 1 + vNi(2) * 1 ) ./ vNi(3);
    vec_1 = [1, 1, x_val];
    vec_1 = vec_1 ./ norm( vec_1 );
    vec_2 = cross(vNi, vec_1);
    vec_2 = vec_2 ./ norm( vec_2 );
    vec_3D_sample = [vec_1; vec_2; vNi];

    % controlled points 
%     p1 = pNi - vec_1 .* scale_plane;
%     p2 = pNi + vec_1 .* scale_plane;
%     p3 = pNi + vec_2 .* scale_plane;
    p1 = pNi - vec_1 .* 0.5 * scale_plane - vec_2 .* 0.5 * scale_plane;
    p2 = pNi + vec_1 .* 0.5 * scale_plane - vec_2 .* 0.5 * scale_plane;
    p3 = pNi + vec_1 .* 0.5 * scale_plane + vec_2 .* 0.5 * scale_plane;
    p4 = pNi - vec_1 .* 0.5 * scale_plane + vec_2 .* 0.5 * scale_plane;
    pts_3D_sample = [p1; p2; p3; p4];

    % fit a surface with the mesh-grid data 
    % 3D non-colinear points
    x_plane = [p1(1) p2(1) p3(1)];  
    y_plane = [p1(2) p2(2) p3(2)];
    z_plane = [p1(3) p2(3) p3(3)];

    % three-controlled-points
    xLim = [min(x_plane) max(x_plane)];
    zLim = [min(z_plane) max(z_plane)];
    [X, Z] = meshgrid(xLim, zLim);

    A = vNi(1); 
    B = vNi(2); 
    C = vNi(3);
    D = -dot(vNi, p1);
    Y = (A * X + C * Z + D)/ (-B);
    reOrder = [1 2 4 3];
    
    % redefine the (X, Y, Z) as a point list (for this problem)
    reOrder = [1 2 3 4];
    X = [p1(1), p2(1), p3(1), p4(1)];
    Y = [p1(2), p2(2), p3(2), p4(2)];
    Z = [p1(3), p2(3), p3(3), p4(3)];

    % save the data for the plane
    data_plane = [];
    data_plane.pts_3D_sample = pts_3D_sample;
    data_plane.vec_3D_sample = vec_3D_sample;
    data_plane.index_order = reOrder;
    data_plane.X_plane = X;
    data_plane.Y_plane = Y;
    data_plane.Z_plane = Z;
    
end
