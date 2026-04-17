function features = feature_extraction(seg_mask, img_gray)
%   Inputs:
%       seg_mask - Binary segmentation mask (H x W logical)
%       img_gray - Original pre-processed grayscale double image (H x W)
%   Output:
%       features - 1-D row vector of 7 statistical features:
%                  [mean, std, entropy, 3rd_moment, 4th_moment,
%                   smoothness, uniformity]

    % Extract pixel values inside the mask
    pixels = double(img_gray(seg_mask));
    if isempty(pixels)
        pixels = double(img_gray(:));
    end

    % Basic first-order statistics
    mu = mean(pixels);
    sigma = std(pixels);
    ent = entropy_val(pixels);
    % Higher-order moments (normalised)
    n3 = mean((pixels - mu).^3) / (sigma^3 + eps);
    n4 = mean((pixels - mu).^4) / (sigma^4 + eps);

    % Smoothness and Uniformity from histogram
    [hcounts, ~] = histcounts(pixels, 64, 'Normalization', 'probability');
    hcounts(hcounts == 0) = [];
    uniformity = sum(hcounts.^2);
    smoothness = 1 - 1 / (1 + sigma^2);

    features = [mu, sigma, ent, n3, n4, smoothness, uniformity];
end

% Shannon entropy from pixel array
function H = entropy_val(pixels)
    [counts, ~] = histcounts(pixels, 256, 'Normalization', 'probability');
    counts(counts == 0) = [];
    H = -sum(counts .* log2(counts));
end
