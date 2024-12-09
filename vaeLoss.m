function loss = vaeLoss(X, latentDim, clusterCenters, clusterFreq, encoder, decoder)
    % Encoder output: mean (mu) and log-variance (log_sigma)
    mu_logsigma = predict(encoder, X);
    mu = mu_logsigma(:, 1:latentDim); % Latent mean
    log_sigma = mu_logsigma(:, latentDim+1:end); % Log variance
    
    % Sample z from the latent distribution
    z = sampleLatent(mu, log_sigma);
    
    % Decode z to get reconstructed image
    X_recon = predict(decoder, z);
    
    % Reconstruction loss
    rec_loss = mse(X_recon, X);
    
    % KL Divergence Loss (Regularization)
    kl_loss = -0.5 * sum(1 + log_sigma - mu.^2 - exp(log_sigma), 2);
    
    % Clustering Loss
    L_clus = clusteringLoss(z, clusterCenters, clusterFreq);
    
    % Total loss (Reconstruction + KL Divergence + Clustering)
    loss = mean(rec_loss + kl_loss + L_clus);
end