function [model, featureMask, mu_train, std_train] = train_model(datasetPath, kernelType, useGA)
% TRAIN_MODEL Build and save a complete brain-tumour classification pipeline.
%
%   [model, featureMask, mu_train, std_train] = 
%       TRAIN_MODEL(datasetPath, kernelType, useGA)
%
%   Inputs:
%       datasetPath - Absolute path to the dataset root that contains Training/ and Testing/ sub-folders.
%       kernelType  - SVM kernel: 'rbf' (default) | 'linear' | 'polynomial' | 'quadratic'
%       useGA       - Logical; true = run Genetic Algorithm for feature selection (slower). Default: false.
%
%   Outputs:
%       model        - Trained ECOC SVM model
%       featureMask  - Logical feature-selection mask (1 x 11)
%       mu_train     - Training-set feature means  (for z-score normalisation)
%       std_train    - Training-set feature stds
%
%   The function also saves 'brain_tumor_model.mat' in datasetPath.

    if nargin < 2 || isempty(kernelType)
        kernelType = 'rbf';
    end 
    
    if nargin < 3 || isempty(useGA)
        useGA = false;
    end

    fprintf('[train_model] Loading training data from: %s\n', datasetPath);
    [X_train, y_train, classMap] = loadDataset(fullfile(datasetPath, 'Training'));
    [X_test, y_test, ~] = loadDataset(fullfile(datasetPath, 'Testing'), classMap);

    % Feature mask(all features unless GA is requested) 
    numFeatures = size(X_train, 2);
    if useGA
        fprintf('[train_model] Running Genetic Algorithm for feature selection...\n');
        featureMask = genetic_feature_selection(X_train, y_train, numFeatures);
    else 
        featureMask = true(1, numFeatures);
    end

    X_tr = X_train(:, featureMask);
    X_te = X_test(:, featureMask);

    % Z-score normalisation(fit on training set only) 
    mu_train = mean(X_tr);
    std_train = std(X_tr);
    std_train(std_train == 0) = 1;
    
    X_tr_norm = (X_tr - mu_train) ./ std_train;
    X_te_norm = (X_te - mu_train) ./ std_train;

    fprintf('[train_model] Training SVM with kernel: %s\n', kernelType);
    results = svm_classifier(X_tr_norm, y_train, X_te_norm, y_test, kernelType);
    model      = results.model;       % cell array of binary fitcsvm models
    classNames = results.classNames;  % class name order used by the OVA models

    % Report 
    m = results.metrics;
    fprintf('\n=== Evaluation on Test Set ===\n');
    fprintf('  Accuracy    : %.2f%%\n', m.accuracy * 100);
    fprintf('  Sensitivity : %.2f%%\n', m.sensitivity * 100);
    fprintf('  Specificity : %.2f%%\n', m.specificity * 100);
    fprintf('  Precision   : %.2f%%\n', m.precision * 100);
    fprintf('  F1 Score    : %.2f%%\n', m.f1 * 100);
    fprintf('\nConfusion Matrix:\n');
    disp(m.confMat);

    % Save artefact 
    savePath = fullfile(datasetPath, 'brain_tumor_model.mat');
    save(savePath, 'model', 'classNames', 'featureMask', 'mu_train', 'std_train', 'classMap');
    fprintf('[train_model] Model saved to: %s\n', savePath);
end

% =========================================================================
% Load all images from a directory tree, extract features, return table 
% =========================================================================
function [X, y, classMap] = loadDataset(rootDir, classMap) 
    folders = dir(rootDir);
    folders = folders([folders.isdir] & ~startsWith({folders.name}, '.'));

    if nargin < 2 || isempty(classMap) 
        classNames = {folders.name};
        classMap = containers.Map(classNames, 1:numel(classNames));
    end

    X = [];
    y = [];

    for fi = 1:numel(folders) 
        className = folders(fi).name;
        classLabel = classMap(className);
        imgDir = fullfile(rootDir, className);
        imgFiles = [dir(fullfile(imgDir, '*.jpg')); dir(fullfile(imgDir, '*.jpeg')); dir(fullfile(imgDir, '*.png'))];

        fprintf('  [%s] Processing %d images...\n', className, numel(imgFiles));

        for ii = 1:numel(imgFiles) 
            imgPath = fullfile(imgDir, imgFiles(ii).name);
            feat = extractFeaturesFromFile(imgPath);
            X = [X; feat]; %#ok<AGROW>
            y = [y; classLabel]; %#ok<AGROW>
        end 
    end

    y = categorical(y, cell2mat(values(classMap)), keys(classMap));
end

% =========================================================================
%  Full pipeline for one image file → feature vector
% =========================================================================
function feat = extractFeaturesFromFile(imgPath)
    img = imread(imgPath);
    if size(img, 3) == 3 
        img = rgb2gray(img);
    end 
    img = imresize(img, [256 256]);

    [ ~, ~, img_med ] = preprocessing(img);
    [ seg_em, ~, ~, ~] = segmentation(img_med);
    feat = feature_extraction(seg_em, img_med);
end
