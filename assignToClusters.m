% Function to Assign Features to Clusters
function clusterAssignments = assignToClusters(features, clusterCenters)
    % Calculate the Euclidean distance from each feature to all cluster centers
    dist = pdist2(features, clusterCenters, 'euclidean');
    
    % Assign each feature to the nearest cluster
    [~, clusterAssignments] = min(dist, [], 2);  % Nearest cluster center
end
