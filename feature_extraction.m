function features = feature_extraction(seg_mask, img_gray)
% FEATURE_EXTRACTION Compute a statistical feature vector from a segmented MRI region.
%
%   features = FEATURE_EXTRACTION(seg_mask, img_gray)
%
%   Inputs:
%       seg_mask - Binary segmentation mask (H x W logical)
%       img_gray - Original pre-processed grayscale double image (H x W)
%   Output:
%       features - 1-D row vector of 11 statistical features:
%                  [mean, std, entropy, 3rd_moment, 4th_moment,
%                   energy, contrast, IDM, correlation, 
%                   smoothness, uniformity]

    % Extract pixel values inside the mask
    pixels = img_gray(seg_mask);
    if isempty(pixels)
        pixels = img_gray(:);
    end

    % ------------------------------------------------------------------ %
    % 1. Basic first-order statistics
    % ------------------------------------------------------------------ %
    mu = mean(pixels);
    sigma = std(pixels);

    % Entropy
    ent = entropy_val(pixels);

    % Higher-order moments (normalised)
    n3 = mean((pixels - mu).^3) / (sigma^3 + eps);
    n4 = mean((pixels - mu).^4) / (sigma^4 + eps);

    % ------------------------------------------------------------------ %
    % 2. GLCM-based texture features
    % ------------------------------------------------------------------ %
    img_uint8 = uint8(img_gray * 255);
    masked_img = img_uint8;
    masked_img(~seg_mask) = 0;

    glcm = graycomatrix(masked_img, 'NumLevels', 64, 'Offset', [0 1; -1 1; -1 0; -1 -1], 'Symmetric', true);
    glcm = mean(glcm, 3); % average over 4 directions
    glcm = glcm / (sum(glcm(:)) + eps);

    [rows, cols] = size(glcm);
    [I, J] = meshgrid(1:rows, 1:cols);
    I = I'; J = J';

    energy = sum(glcm(:).^2);
    contrast = sum(((I(:) - J(:)).^2) .* glcm(:));
    IDM = sum(glcm(:) ./ (1 + (I(:) - J(:)).^2));

    mu_i = sum(I(:) .* glcm(:));
    mu_j = sum(J(:) .* glcm(:));
    sig_i = sqrt(sum(((I(:) - mu_i).^2) .* glcm(:)) + eps);
    sig_j = sqrt(sum(((J(:) - mu_j).^2) .* glcm(:)) + eps);
    corr = sum(((I(:) - mu_i) .* (J(:) - mu_j) .* glcm(:))) / (sig_i * sig_j);

    % ------------------------------------------------------------------ %
    % 3. Smoothness and Uniformity from histogram
    % ------------------------------------------------------------------ %
    [hcounts, ~] = histcounts(pixels, 64, 'Normalization', 'probability');
    hcounts(hcounts == 0) = [];
    uniformity = sum(hcounts.^2);
    smoothness = 1 - 1 / (1 + sigma^2);

    % ------------------------------------------------------------------ %
    % Assemble output vector
    % ------------------------------------------------------------------ %
    features = [mu, sigma, ent, n3, n4, energy, contrast, IDM, corr, smoothness, uniformity];
end

% =========================================================================
% Local helper: Shannon entropy from pixel array
% =========================================================================
function H = entropy_val(pixels)
    [counts, ~] = histcounts(pixels, 256, 'Normalization', 'probability');
    counts(counts == 0) = [];
    H = -sum(counts .* log2(counts));
end
