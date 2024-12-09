function testModel(testImgFolder, encoder, clusterCenters)
    % Load the test images
    imgSets = imageDatastore(testImgFolder);
    numTestImages = numel(imgSets.Files);
    fprintf('No. of test images = %d\n', numTestImages);
    
    % Prepare arrays to hold the feature vectors and true labels (if available)
    testFeatures = [];
    trueLabels = [];  % ground truth labels if available

    for i = 1:numTestImages
        img = readimage(imgSets, i);
        img = imresize(img, [32, 32]);  % Resize to input size
        
        % Convert to RGB if grayscale
        if size(img, 3) == 1
            img = repmat(img, [1, 1, 3]);  % Convert to RGB
        end
        
        img = single(img) / 255;  % Normalize image to [0, 1]

        % Extract Informative Patches
        [patches, ~] = extractInformativePatches(img, [32, 32, 3]);

        % Extract features using pretrained VAE encoder
        features = [];
        for j = 1:size(patches, 1)
            patch = patches(j, :);
            patch = reshape(patch, [32, 32, 3]);  % Reshape to [32, 32, 3]

            % Get latent feature vector from pretrained VAE encoder
            latentVec = extractFeaturesUsingPretrainedVAE(patch, encoder);
            features = [features; latentVec];
        end
        
        testFeatures = [testFeatures; features];
    end

    
    % Perform clustering (assign test samples to clusters)
    clusterAssignments = assignToClusters(testFeatures, clusterCenters);

    % Evaluate retrieval accuracy (ground truth labels)
    if ~isempty(trueLabels)
        accuracy = calculateAccuracy(clusterAssignments, trueLabels);
        fprintf('Test Accuracy: %.4f\n', accuracy);
    else
        fprintf('Test clustering complete, no true labels provided.\n');
    end
end