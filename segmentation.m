function [seg_em, seg_levelset, seg_watershed, seg_threshold] = segmentation(img_filtered)
% SEGMENTATION Segment a pre-processed MRI image using multiple algorithms.
%
%   [seg_em, seg_levelset, seg_watershed, seg_threshold] = SEGMENTATION(img_filtered)
%
%   Input:
%       img_filtered  - Pre-processed grayscale double image in [0, 1]
%   Output:
%       seg_em        - EM-based segmentation (binary mask of tumour region)
%       seg_levelset  - Level-set refined contour (binary mask)
%       seg_watershed - Watershed segmentation (label matrix, uint32)
%       seg_threshold - Gray-level threshold segmentation (binary mask)

    if size(img_filtered, 3) == 3
        img_filtered = rgb2gray(img_filtered);
    end
    img_d = im2double(img_filtered);

    % --- 1. EM segmentation (Gaussian Mixture Model, K=3 tissues) ---
    seg_em = emSegmentation(img_d, 3);

    % --- 2. Threshold segmentation (Otsu on EM foreground) ---
    level = graythresh(img_d);
    seg_threshold = imbinarize(img_d, level);
    seg_threshold = imfill(seg_threshold, 'holes');
    seg_threshold = bwareaopen(seg_threshold, 50);

    % --- 3. Level-set (Chan-Vese active contour starting from Otsu mask) ---
    seg_levelset = levelSetSegmentation(img_d, seg_threshold);

    % --- 4. Watershed segmentation ---
    seg_watershed = watershedSegmentation(img_d);
end

% =========================================================================
% EM segmentation via Gaussian Mixture Model
% =========================================================================
function mask = emSegmentation(img, K)
    pixels = img(:);
    gm = fitgmdist(pixels, K, 'RegularizationValue', 1e-5);

    [~, ~, post] = cluster(gm, pixels);

    % pick the cluster whose mean is highest (brightest -> tumour candidate)
    [~, tumorCluster] = max(gm.mu);
    tumourPost = post(:, tumorCluster);
    mask = reshape(tumourPost > 0.5, size(img));

    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 30);
end

% =========================================================================
% Level-set (Chan-Vese via activecontour)
% =========================================================================
function mask = levelSetSegmentation(img, initMask)
    if ~any(initMask(:))
        mask = initMask;
        return;
    end
    try
        mask = activecontour(img, initMask, 200, 'Chan-Vese', 'ContractionBias', 0.1, 'SmoothFactor', 1);
    catch
        mask = initMask;
    end
    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 30);
end

% =========================================================================
% Watershed segmentation
% =========================================================================
function L = watershedSegmentation(img)
    % gradient magnitude as relief map
    [Gmag, ~] = imgradient(img, 'sobel');
    Gmag = mat2gray(Gmag);

    % suppress over-segmentation with markers
    Gmag_min = imextendedmin(Gmag, 0.02);
    Gmag_mod = imimposemin(Gmag, Gmag_min);

    L = watershed(Gmag_mod);
end
