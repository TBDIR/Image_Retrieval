function L_clus = clusteringLoss(z, clusterCenters, f)
    % z: Latent features of the data (feature points)
    % clusterCenters: Cluster centers in the latent space
    % f: Frequency of each cluster (initialization)

    % Compute the soft assignment of each feature point to clusters
    s = exp(-pdist2(z, clusterCenters, 'squaredeuclidean'));  % Similarity matrix
    sum_s = sum(s, 2);  % Normalize similarities by row
    r = s ./ sum_s;  % Soft assignment probabilities

    % Compute cluster frequencies (how many feature points belong to each cluster)
    f_j = sum(r, 1);  % Frequency of each cluster

    % Scale the similarity matrix and soft assignment matrix by cluster frequencies
    s_scaled = s ./ f_j;
    r_scaled = r ./ f_j;

    % Compute the Kullback-Leibler (KL) divergence between the scaled distributions
    L_clus = sum(sum(s_scaled .* log(s_scaled ./ r_scaled)));  % Clustering loss
end