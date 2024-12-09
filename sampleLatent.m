function z = sampleLatent(mu, logsigma)
    epsilon = randn(size(mu)); % Standard normal noise
    z = mu + exp(logsigma / 2) .* epsilon; % Reparameterization trick
end