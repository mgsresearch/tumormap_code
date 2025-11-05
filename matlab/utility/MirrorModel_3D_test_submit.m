
function [pIi, vIi, vOi, LSi, data_plane] = MirrorModel_3D_test_submit( pSi, vSi, pNi, vNi )

    % argvs:
        % pSi: 3 x 1 
        % vSi: 3 x 1 
        % pNi: 3 x 1 
        % vNi: 3 x 1 
    % returns: 
        % pIi: 3 x 1 
        % vIi: 3 x 1
        % vOi: 3 x 1
        % LSi: 1 x 1
        % data_plane:

    % normalize data format
    pSi = pSi';         % 1 x 3 
    vSi = vSi';         % 1 x 3 
    pNi = pNi';         % 1 x 3 
    vNi = vNi';         % 1 x 3 

    % normalization of the input vector
    % vSi = vSi ./ norm(vSi); 
    
    % normalization of the mirror normal
    % vNi = vNi ./ norm(vNi); 

    % calculate the reflected vector 
    % vS2Ni = pSi - pNi; 
    dN = -dot( vNi, pSi - pNi ); 
    dS =  dot( vNi, vSi ); 
    ratio_NS = dN ./ dS;

    % intersection point 
    pIi = pSi + ratio_NS * vSi; 
    vOi = vSi - 2 * ( dot ( vSi, vNi ) ) * vNi;

    % The length of incidente vector
    LSi = norm( pIi - pSi ); 
    
    % The input vector length
    vIi = pIi - pSi;
    vIi = vIi ./ norm( vIi );
    
    % 3D Plane 
    scale_plane = 0.15;
    [data_plane] = get_Plane_submit(vNi, pNi, scale_plane);  

    % change the data format
    pIi = pIi'; 
    vIi = vIi'; 
    vOi = vOi'; 
    
end


