clc;
clear all;
close all;

% Load Pretrained VAE
load('pretrainedVAE.mat', 'encoder', 'decoder');  % Pretrained VAE model has encoder and decoder

% Load Image Dataset
imgFolder = 'oxford5k1';  % Path to your dataset
imgSets = imageDatastore(imgFolder);
numImages = numel(imgSets.Files);
fprintf('\nNo. of images = %d\n', numImages);

% Image Parameters
inputSize = [32, 32, 3];  % Size of image patches for VAE
latentDim = 64;  % Dimensionality of the latent space
clusterDim = 100; % Number of clusters for DEC

% Initialize Clustering Centers and Frequencies
C = clusterDim;  % Number of clusters
clusterCenters = randn(C, latentDim); % Randomly initialize cluster centers
clusterFreq = ones(1, C); % Initialize cluster frequencies

% Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Training Loop (Clustering + DEC with Pretrained VAE)
for epoch = 1:options.MaxEpochs
    totalLoss = 0;
    
    % Loop through images and perform training
    for i = 1:numImages
        img = readimage(imgSets, i);
        img = imresize(img, inputSize(1:2)); % Resize image to input size
        
        % Convert to RGB if grayscale
        if size(img, 3) == 1
            img = repmat(img, [1, 1, 3]); % Convert to RGB
        end
        
        img = single(img) / 255; % Normalize image to [0, 1]
        
        % Extract Informative Patches
        [patches, patchLocs] = extractInformativePatches(img, inputSize);
        
        % Feature Extraction using Pretrained VAE Encoder
        features = [];
        for j = 1:size(patches, 1)
            patch = patches(j, :);
            patch = reshape(patch, inputSize); % Reshape patch to 32x32x3
            
            % Extract latent vector using pretrained VAE encoder
            latentVec = extractFeaturesUsingPretrainedVAE(patch, encoder);
            features = [features; latentVec];
        end
        
        % Compute clustering loss based on extracted latent features
        loss = clusteringLoss(features, clusterCenters, clusterFreq);
        
        % Update total loss
        totalLoss = totalLoss + loss;
        
        % Update cluster centers based on DEC (clustering)
        clusterCenters = updateClusterCenters(features, clusterCenters, clusterFreq);
    end
    
    % Display loss after each epoch
    fprintf('Epoch %d, Loss: %.4f\n', epoch, totalLoss / numImages);
end

% Save the updated clustering model
save('vaeDEC_Model.mat', 'clusterCenters', 'clusterFreq');
