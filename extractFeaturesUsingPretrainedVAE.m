% Function to Extract Latent Features Using Pretrained VAE Encoder
function latentVec = extractFeaturesUsingPretrainedVAE(img, encoder)
    % Normalize and resize the image as required by the VAE
    img = imresize(img, [32, 32]); % Resize image
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]); % Convert grayscale to RGB if necessary
    end
    img = single(img) / 255;  % Normalize image

    % Extract the latent vector using the pretrained encoder
    latentVec = encoder.predict(img);  % Get the latent representation
end