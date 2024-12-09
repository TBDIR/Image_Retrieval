function clusterCenters = updateClusterCenters(z, clusterCenters, clusterFreq)
    % Update the cluster centers based on the current assignment of feature points to clusters
    % z: Latent features (feature points)
    % clusterCenters: Current cluster centers in the latent space
    % clusterFreq: Frequency of each cluster (not used here, but can be incorporated if needed)

    % Number of clusters
    C = size(clusterCenters, 1);

    % Compute squared Euclidean distance between data points and cluster centers
    dist = pdist2(z, clusterCenters, 'sqeuclidean');

    % Assign each data point to the nearest cluster
    [~, idx] = min(dist, [], 2);  % idx contains the index of the closest cluster for each feature point

    % Update the cluster centers by computing the mean of all feature assigned to each cluster
    for c = 1:C
        clusterCenters(c, :) = mean(z(idx == c, :), 1);  % Update the center for cluster c
    end
end