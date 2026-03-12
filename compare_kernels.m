function compare_kernels(datasetPath)
% COMPARE_KERNELS  Train and evaluate SVMs with all 4 kernels, then plot results.
%
%   compare_kernels(datasetPath)
%
%   Input:
%       datasetPath - Path to the dataset root (contains Training/ and Testing/)
%
%   This function:
%     1. Extracts all features from the dataset (or loads cached features).
%     2. Trains four SVM models (RBF, Linear, Polynomial, Quadratic).
%     3. Displays a bar-chart comparing accuracy, sensitivity and specificity.
%     4. Prints confusion matrices for each kernel.

    kernels = {'rbf', 'linear', 'polynomial', 'quadratic'};
    cacheFile = fullfile(datasetPath, 'feature_cache.mat');

    % ------------------------------------------------------------------ %
    % Feature extraction(or load cache)
    % ------------------------------------------------------------------ %
    if isfile(cacheFile)
        fprintf('[compare_kernels] Loading cached features from: %s\n', cacheFile);
        S = load(cacheFile, 'X_train', 'y_train', 'X_test', 'y_test');
        X_train = S.X_train;
        y_train = S.y_train;
        X_test = S.X_test;
        y_test = S.y_test;
    else 
        fprintf('[compare_kernels] Extracting features (this may take a while)...\n');
        [X_train, y_train] = extractAll(fullfile(datasetPath, 'Training'));
        [X_test, y_test] = extractAll(fullfile(datasetPath, 'Testing'));
        save(cacheFile, 'X_train', 'y_train', 'X_test', 'y_test');
        fprintf('[compare_kernels] Features saved to cache.\n');
    end

    % Z-score normalisation(fit on train) 
    mu = mean(X_train);
    sig = std(X_train);
    sig(sig == 0) = 1;
    X_tr_n = (X_train - mu) ./ sig;
    X_te_n = (X_test - mu) ./ sig;

    % ------------------------------------------------------------------ %
    % Train &evaluate each kernel
    % ------------------------------------------------------------------ %
    nK = numel(kernels);
    acc = zeros(nK, 1);
    sens = zeros(nK, 1);
    spec = zeros(nK, 1);
    prec = zeros(nK, 1);
    f1s = zeros(nK, 1);

    for k = 1:nK 
        fprintf('\n=== Kernel: %s ===\n', kernels{k});
        res = svm_classifier(X_tr_n, y_train, X_te_n, y_test, kernels{k});
        m = res.metrics;
        acc(k) = m.accuracy * 100;
        sens(k) = m.sensitivity * 100;
        spec(k) = m.specificity * 100;
        prec(k) = m.precision * 100;
        f1s(k) = m.f1 * 100;

        fprintf('  Accuracy    : %.2f%%\n', acc(k));
        fprintf('  Sensitivity : %.2f%%\n', sens(k));
        fprintf('  Specificity : %.2f%%\n', spec(k));
        fprintf('  Precision   : %.2f%%\n', prec(k));
        fprintf('  F1 Score    : %.2f%%\n', f1s(k));
        fprintf('  Confusion Matrix:\n');
        disp(m.confMat);
    end

    % ------------------------------------------------------------------ %
    % Plot comparison
    % ------------------------------------------------------------------ %
    figure('Name', 'Kernel Comparison', 'Color', [0.08 0.09 0.12], 'Position', [100 100 900 500]);

    data = [acc, sens, spec, prec, f1s];
    bh = bar(data, 'grouped');

    metricColors = [
        0.20 0.55 0.95; % blue - accuracy 
        0.18 0.78 0.56; % green - sensitivity 
        0.95 0.70 0.20; % amber - specificity 
        0.75 0.40 0.85; % purple - precision 
        0.95 0.42 0.42  % red - f1
    ];  
    for j = 1:5
        bh(j).FaceColor = metricColors(j, :);
        bh(j).EdgeColor = 'none';
    end

    ax = gca;
    ax.Color = [0.11 0.13 0.17];
    ax.XColor = [0.75 0.77 0.82];
    ax.YColor = [0.75 0.77 0.82];
    ax.GridColor = [0.3 0.32 0.38];
    ax.GridAlpha = 0.4;
    ax.YGrid = 'on';
    ax.FontSize = 11;
    ax.XTickLabel = kernels;
    ax.XTickLabelRotation = 0;

    ylim([0 105]);
    ylabel('Score (%)', 'Color', [0.75 0.77 0.82]);
    title('SVM Kernel Comparison', 'Color', [0.95 0.95 0.97], 'FontSize', 14);
    legend({'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1'}, ...
        'TextColor', [0.75 0.77 0.82], 'Color', [0.11 0.13 0.17], ...
        'EdgeColor', [0.3 0.32 0.38], 'Location', 'southeast');
end

% =========================================================================
% Extract features from all images in a folder tree
% =========================================================================
function [X, y] = extractAll(rootDir) 
    folders = dir(rootDir);
    folders = folders([folders.isdir] & ~startsWith({folders.name}, '.'));

    classNames = {folders.name};
    classMap = containers.Map(classNames, 1:numel(classNames));

    X = [];
    y = [];

    for fi = 1:numel(folders) 
        className = folders(fi).name;
        classLabel = classMap(className);
        imgDir = fullfile(rootDir, className);
        imgFiles = [dir(fullfile(imgDir, '*.jpg')); dir(fullfile(imgDir, '*.jpeg')); dir(fullfile(imgDir, '*.png'))];

        fprintf('  [%s] %d images\n', className, numel(imgFiles));

        for ii = 1:numel(imgFiles) 
            imgPath = fullfile(imgDir, imgFiles(ii).name);
            try 
                img = imread(imgPath);
                if size(img, 3) == 3
                    img = rgb2gray(img);
                end 
                img = imresize(img, [256 256]);
                
                [ ~, ~, img_med ] = preprocessing(img);
                [ seg_em, ~, ~, ~] = segmentation(img_med);
                feat = feature_extraction(seg_em, img_med);
                
                X = [X; feat]; %#ok<AGROW> 
                y = [y; classLabel]; %#ok<AGROW> 
            catch 
                % skip corrupted images 
            end 
        end 
    end

    y = categorical(y, cell2mat(values(classMap)), keys(classMap));
end
