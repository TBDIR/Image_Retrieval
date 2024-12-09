clc;
clear all;
close all;

% Image Folder Location
imgFolder = 'oxford5k1'; % Or 'Coil_DB_1129', or 'Paris1'
imgSets = imageDatastore(imgFolder);
numImages = numel(imgSets.Files);
fprintf('\nNo. of images = %d\n', numImages);

% Initialize feature matrix
all_fv = []; % Only initialize once

% Initialize VAE model (make sure you load your trained VAE model here)
% Example: load('vaeModel.mat'); % Load your pre-trained VAE model

% Initialize counter k
k = 1;

% Loop through each image
for j = 1:numImages
    disp(j);
    % Read and preprocess the image
    img = readimage(imgSets, j);
    img = imresize(img, [1024, 1024]);
    grayImg = rgb2gray(img); % Convert to grayscale

    % Grid for patches
    [height, width] = size(grayImg);
    gridstep = 32;
    gridX = 1:gridstep:width - 32;
    gridY = 1:gridstep:height - 32;
    [x_value, y_value] = meshgrid(gridX, gridY);
    gridlocations = [y_value(:), x_value(:)];

    % Entropy-based thresholding
    image_entropy = [];
    max_patch_entropy = realmin;

    for loc = 1:size(gridlocations, 1)
        patch = grayImg(gridlocations(loc, 1):(gridlocations(loc, 1) + 31), ...
                        gridlocations(loc, 2):(gridlocations(loc, 2) + 31));
        patch_entropy = entropy(patch);
        max_patch_entropy = max(max_patch_entropy, patch_entropy);
        image_entropy = [image_entropy; patch_entropy];
    end

    % Normalize and threshold entropy
    vis_entropy = round(255 * (image_entropy / max_patch_entropy));
    threshold = mean(vis_entropy) + std(vis_entropy);

    % Filter patches based on threshold
    new_gridlocations = gridlocations(vis_entropy > threshold, :);

    % Extract Features Using Encoder (for each patch)
    featuresTrain = [];
    for loc = 1:size(new_gridlocations, 1)
        patch = grayImg(new_gridlocations(loc, 1):(new_gridlocations(loc, 1) + 31), ...
                        new_gridlocations(loc, 2):(new_gridlocations(loc, 2) + 31));
        patch = imresize(patch, [32, 32]); % Resize patch for VAE input
        
        % Convert grayscale patch to RGB (if required)
        train_img = repmat(patch, [1, 1, 3]); % Convert to RGB by replicating the gray values
        train_img = dlarray(single(train_img), 'SSC');

        % Pass through the VAE encoder to get latent features
        latentFeatures = predict(vae, train_img);  % Assuming 'vae' is your trained VAE model
        featuresTrain = [featuresTrain; extractdata(latentFeatures)'];
    end

    % Remove rows with all zeros (if any)
    fv = featuresTrain(any(featuresTrain, 2), :);

    % Append to all_fv
    if k == 1
        all_fv = fv;
    else
        all_fv = [all_fv; fv];
    end
    k = k + 1;
end

% Display final feature matrix size
disp(size(all_fv));