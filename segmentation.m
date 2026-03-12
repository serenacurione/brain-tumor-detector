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
% EM segmentation via Gaussian Mixture Model (custom, no toolbox needed)
% =========================================================================
function mask = emSegmentation(img, K)
    pixels = double(img(:));
    N = numel(pixels);

    % --- Initialisation (k-means style: evenly spaced means) ---
    mu  = linspace(min(pixels), max(pixels), K);
    sig = ones(1, K) * var(pixels) / K + 1e-6;   % variance per component
    pi_ = ones(1, K) / K;                          % mixing weights

    maxIter = 100;
    tol     = 1e-6;
    logLikPrev = -Inf;

    gamma = zeros(N, K);   % responsibilities

    for iter = 1:maxIter
        % --- E-step: compute responsibilities ---
        for k = 1:K
            gamma(:, k) = pi_(k) * gaussPdf(pixels, mu(k), sig(k));
        end
        sumGamma = sum(gamma, 2) + 1e-300;   % avoid /0
        gamma    = gamma ./ sumGamma;

        % --- Log-likelihood (for convergence check) ---
        logLik = sum(log(sumGamma));
        if abs(logLik - logLikPrev) < tol
            break;
        end
        logLikPrev = logLik;

        % --- M-step: update parameters ---
        Nk = sum(gamma, 1) + 1e-10;          % effective count per cluster
        for k = 1:K
            mu(k)  = (gamma(:,k)' * pixels) / Nk(k);
            sig(k) = (gamma(:,k)' * (pixels - mu(k)).^2) / Nk(k) + 1e-6;
            pi_(k) = Nk(k) / N;
        end
    end

    % pick the cluster with the highest mean (brightest → tumour candidate)
    [~, tumorCluster] = max(mu);
    tumourPost = gamma(:, tumorCluster);
    mask = reshape(tumourPost > 0.5, size(img));

    mask = imfill(mask, 'holes');
    mask = bwareaopen(mask, 30);
end

% Gaussian PDF helper
function p = gaussPdf(x, mu, sigma2)
    p = exp(-0.5 * (x - mu).^2 / sigma2) / sqrt(2 * pi * sigma2);
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
