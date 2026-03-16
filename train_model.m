function [model, featureMask, mu_train, std_train] = train_model(datasetPath, kernelType, varargin)
% TRAIN_MODEL Build and save a complete brain-tumour classification pipeline.
%
%   [model, featureMask, mu_train, std_train] = 
%       TRAIN_MODEL(datasetPath, kernelType)
%   [model, featureMask, mu_train, std_train] = 
%       TRAIN_MODEL(datasetPath, kernelType, 'GridSearch', true, ...)
%
%   Inputs:
%       datasetPath - Absolute path to the dataset root that contains
%                     Training/ and Testing/ sub-folders.
%       kernelType  - SVM kernel: 'rbf' (default) | 'linear' |
%                    'polynomial' | 'quadratic'
%
%   Optional Name-Value Pairs (passed through to svm_grid_search):
%       'GridSearch'   - Run grid search before final training (default: false)
%       'KFolds'       - Number of CV folds for grid search   (default: 5)
%       'CValues'      - BoxConstraint grid                   (default: logspace(-2,3,6))
%       'GammaValues'  - KernelScale grid for rbf/poly        (default: logspace(-3,2,6))
%
%   Outputs:
%       model        - Trained ECOC SVM model
%       featureMask  - Logical feature-selection mask (1 x F)
%       mu_train     - Training-set feature means  (for z-score normalisation)
%       std_train    - Training-set feature stds
%
%   The function also saves 'brain_tumor_model.mat' in datasetPath.

    if nargin < 2 || isempty(kernelType)
        kernelType = 'rbf';
    end

    % ------------------------------------------------------------------
    % Parse optional grid-search arguments
    % ------------------------------------------------------------------
    p = inputParser();
    p.addParameter('GridSearch',   false,              @islogical);
    p.addParameter('KFolds',       5,                  @(x) isnumeric(x) && x >= 2);
    p.addParameter('CValues',      logspace(-2, 3, 6), @isnumeric);
    p.addParameter('GammaValues',  logspace(-3, 2, 6), @isnumeric);
    p.parse(varargin{:});

    runGridSearch = p.Results.GridSearch;
    kFolds        = p.Results.KFolds;
    C_grid        = p.Results.CValues;
    gamma_grid    = p.Results.GammaValues;

    % ------------------------------------------------------------------
    % Load dataset
    % ------------------------------------------------------------------
    fprintf('[train_model] Loading training data from: %s\n', datasetPath);
    [X_train, y_train, classMap] = loadDataset(fullfile(datasetPath, 'Training'));
    [X_test, y_test, ~] = loadDataset(fullfile(datasetPath, 'Testing'), classMap);

    % Use all features
    numFeatures = size(X_train, 2);
    featureMask = true(1, numFeatures);

    X_tr = X_train(:, featureMask);
    X_te = X_test(:, featureMask);

    % Z-score normalisation (fit on training set only)
    mu_train  = mean(X_tr);
    std_train = std(X_tr);
    std_train(std_train == 0) = 1;

    X_tr_norm = (X_tr - mu_train) ./ std_train;
    X_te_norm = (X_te - mu_train) ./ std_train;

    % ------------------------------------------------------------------
    % Optional Grid Search + Cross-Validation
    % ------------------------------------------------------------------
    bestC     = 1;      % default hyperparameters
    bestGamma = NaN;

    if runGridSearch
        fprintf('\n[train_model] Running Grid Search + %d-Fold Cross-Validation...\n', kFolds);

        [bestParams, gsResults] = svm_grid_search(X_tr_norm, y_train, kernelType, ...
            'KFolds',       kFolds, ...
            'CValues',      C_grid, ...
            'GammaValues',  gamma_grid, ...
            'Verbose',      true);

        kernelType = bestParams.kernelType;
        bestC      = bestParams.C;
        bestGamma  = bestParams.gamma;

        fprintf('\n[train_model] Grid Search complete.\n');
        fprintf('  Best Kernel= %s\n', upper(kernelType));
        fprintf('  Best C     = %.4g\n', bestC);
        if ~isnan(bestGamma)
            fprintf('  Best Gamma = %.4g\n', bestGamma);
        end
        fprintf('  Best CV Accuracy = %.2f%%\n\n', bestParams.accuracy * 100);

        % Save grid search results alongside model
        gsPath = fullfile(datasetPath, 'grid_search_results.mat');
        save(gsPath, 'gsResults', 'bestParams');
        fprintf('[train_model] Grid search results saved to: %s\n', gsPath);

        % Print the full CV accuracy tables for all evaluated kernels
        for kIdx = 1:numel(gsResults)
            fprintf('\n--- Cross-Validation Accuracy Grid (%s kernel) ---\n', gsResults(kIdx).kernelType);
            printGridTable(gsResults(kIdx));
        end
    elseif ischar(kernelType) && strcmpi(kernelType, 'auto')
        % Fallback if 'auto' is used but grid search is false
        fprintf('[train_model] WARNING: ''auto'' kernel used without GridSearch. Defaulting to ''rbf''.\n');
        kernelType = 'rbf';
    end

    % ------------------------------------------------------------------
    % Final training with (optimal) hyperparameters
    % ------------------------------------------------------------------
    fprintf('\n[train_model] Training final SVM with kernel: %s\n', kernelType);

    if ~isnan(bestGamma)
        results = svm_classifier(X_tr_norm, y_train, X_te_norm, y_test, kernelType, ...
            'BoxConstraint', bestC, ...
            'KernelScale',   bestGamma);
    else
        results = svm_classifier(X_tr_norm, y_train, X_te_norm, y_test, kernelType, ...
            'BoxConstraint', bestC);
    end

    model      = results.model;
    classNames = results.classNames;

    % ------------------------------------------------------------------
    % Report
    % ------------------------------------------------------------------
    m = results.metrics;
    fprintf('\n=== Evaluation on Test Set ===\n');
    fprintf('  Accuracy    : %.2f%%\n', m.accuracy * 100);
    fprintf('  Sensitivity : %.2f%%\n', m.sensitivity * 100);
    fprintf('  Specificity : %.2f%%\n', m.specificity * 100);
    fprintf('  Precision   : %.2f%%\n', m.precision * 100);
    fprintf('  F1 Score    : %.2f%%\n', m.f1 * 100);
    fprintf('\nConfusion Matrix:\n');
    disp(m.confMat);

    % ------------------------------------------------------------------
    % Save artefact
    % ------------------------------------------------------------------
    savePath = fullfile(datasetPath, 'brain_tumor_model.mat');
    save(savePath, 'model', 'classNames', 'featureMask', 'mu_train', 'std_train', 'classMap');
    fprintf('[train_model] Model saved to: %s\n', savePath);
end

% =========================================================================
%  Pretty-print the CV accuracy grid
% =========================================================================
function printGridTable(gsResults)
    C_grid     = gsResults.C_grid;
    gamma_grid = gsResults.gamma_grid;
    cvAcc      = gsResults.cvAccuracy * 100;   % convert to %
    bestIdx    = gsResults.bestIdx;

    hasGamma = ~(numel(gamma_grid) == 1 && isnan(gamma_grid(1)));

    if hasGamma
        % Header
        header = sprintf('%-10s', 'C \\ Gamma');
        for gi = 1:numel(gamma_grid)
            header = [header, sprintf('%-10.3g', gamma_grid(gi))]; %#ok<AGROW>
        end
        fprintf('%s\n', header);
        fprintf('%s\n', repmat('-', 1, length(header)));

        for ci = 1:numel(C_grid)
            row = sprintf('%-10.3g', C_grid(ci));
            for gi = 1:numel(gamma_grid)
                val = cvAcc(ci, gi);
                if ci == bestIdx(1) && gi == bestIdx(2)
                    row = [row, sprintf('[%6.2f%%] ', val)]; %#ok<AGROW>
                else
                    row = [row, sprintf(' %6.2f%%  ', val)]; %#ok<AGROW>
                end
            end
            fprintf('%s\n', row);
        end
    else
        fprintf('%-10s %-12s\n', 'C', 'CV Acc (%)');
        fprintf('%s\n', repmat('-', 1, 24));
        for ci = 1:numel(C_grid)
            marker = '';
            if ci == bestIdx(1)
                marker = ' <-- BEST';
            end
            fprintf('%-10.3g %-12.2f%s\n', C_grid(ci), cvAcc(ci,1), marker);
        end
    end
    fprintf('\n(Values in brackets = best configuration)\n\n');
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
%  Full pipeline for one image file -> feature vector
% =========================================================================
function feat = extractFeaturesFromFile(imgPath)
    img = imread(imgPath);
    if size(img, 3) == 3 
        img = rgb2gray(img);
    end 
    img = imresize(img, [256 256]);

    [ ~, ~, img_med ] = preprocessing(img);
    [ seg_em, ~] = segmentation(img_med);
    feat = feature_extraction(seg_em, img_med);
end
