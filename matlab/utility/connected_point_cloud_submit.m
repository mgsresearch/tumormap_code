function [ cell_pc ] = connected_point_cloud_submit( pt_use, num_of_knn, scale_dis, flag_figure )

    % connected surface point cloud 
    % the connected components 
    % variables: 
    % 1. number of knn
    % 2. distance of scale

    % % initialize the figure 
    % figure(flag_figure);clf;

    % pt_use = data_icp;
    x_0 = pt_use;
    y_0 = pt_use(1,:);
    [n_0, d_0] = knnsearch(x_0, y_0, 'k', num_of_knn, 'distance', 'euclidean');
    
    x = pt_use;         % The main data set
    y = pt_use(n_0,:);  % The closest n_0 points
    result = [n_0(:)];  % The current result of the cloest points index 
    Cell = {};          % The Cell -- final results 
    flag_group = 0;     
    data_viz = [y];
    count = 0;
    
    for i = 1: 500
    
        %count = count + 1
        % i
    
        [n, d] = knnsearch(x, y, 'k', num_of_knn, 'distance', 'euclidean');
    
        thres_remove = scale_dis; 
        idx_out = find(d(:) > thres_remove);    % index that needs to be removed out 
        
        % index to tbe kept for each iteration
        n_iter = n(:);
        n_iter(idx_out) = [];
        n = n_iter;
    
        n_idx = unique(n(:));
    
        % storage the results
        result = [result; n_idx];
    
        y = pt_use(n_idx, :, :);
        data_viz = [data_viz;y];
    
        % Determine the next x and y
        idx_remove = n_idx;
        pt_use(idx_remove,:) = [];
    
        x = pt_use;
    
        % % viz the result
        % figure(flag_figure);hold on; 
        % xlabel('x');ylabel('y');zlabel('z');
        % pcshow(y,'r');
    
        % stop
        if length(y) == 0 & length(x) == 0
            flag_group = flag_group + 1;
            Cell{flag_group} = data_viz;
            break;
        end
    
        if length(y) == 0 & length(x) ~= 0
            flag_group = flag_group + 1;
            Cell{flag_group} = data_viz;
            data_viz = [];
            y = pt_use(1,:);
            data_viz = [data_viz;y];
        end
    
    end
    
    % sort the clusters 
    size_of_cluster = [];
    for i = 1: length(Cell) 
        res_viz = Cell{i};  
        size_of_cluster = [size_of_cluster;length(res_viz)];
    end
    [size_of_cluster_sort, idx_sort] = sort(size_of_cluster, "descend"); 
    
    % remove the other point cloud
    [L_max, idx_max] = max(size_of_cluster);   
    if length(Cell) == 0
        data_use = x_0;
    else
        data_use = Cell{idx_max};
    end
    data = data_use;
    
    cell_pc = Cell;

end
