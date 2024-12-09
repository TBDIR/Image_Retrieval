% Function to Extract Informative Patches from an Image
function [patches, patchLocs] = extractInformativePatches(img, patchSize)
    % Convert image to grayscale
    grayImg = rgb2gray(img);
    
    % Parameters
    [height, width] = size(grayImg);
    gridstep = patchSize(1);  % Step size for grid, based on patch size
    gridX = 1:gridstep:width - patchSize(2);  % Adjust based on patch width
    gridY = 1:gridstep:height - patchSize(1);  % Adjust based on patch height
    [x_value, y_value] = meshgrid(gridX, gridY);
    gridlocations = [y_value(:), x_value(:)];
    
    % Entropy-based thresholding for patch selection
    image_entropy = [];
    max_patch_entropy = realmin;
    for loc = 1:size(gridlocations, 1)
        % Extract patch based on the patch size
        patch = grayImg(gridlocations(loc, 1):(gridlocations(loc, 1) + patchSize(1) - 1), ...
                        gridlocations(loc, 2):(gridlocations(loc, 2) + patchSize(2) - 1));
        patch_entropy = entropy(patch);
        max_patch_entropy = max(max_patch_entropy, patch_entropy);
        image_entropy = [image_entropy; patch_entropy];
    end
    
    % Normalize entropy and set threshold
    vis_entropy = round(255 * (image_entropy / max_patch_entropy));
    threshold = mean(vis_entropy) + std(vis_entropy);
    
    % Select informative patches based on the threshold
    new_gridlocations = gridlocations(vis_entropy > threshold, :);
    
    % Extract the selected patches from the original image
    patches = [];
    patchLocs = new_gridlocations;
    for loc = 1:size(new_gridlocations, 1)
        % Extract patch based on the patch size
        patch = img(new_gridlocations(loc, 1):(new_gridlocations(loc, 1) + patchSize(1) - 1), ...
                    new_gridlocations(loc, 2):(new_gridlocations(loc, 2) + patchSize(2) - 1), :);
        patches = [patches; patch(:)'];  % Reshape each patch to a vector and append
    end
end