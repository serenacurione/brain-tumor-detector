function [model] = train_model(datasetPath, varargin)
%   Inputs:
%       datasetPath - Absolute path to the dataset root that contains
%                     Training/ and Testing/ sub-folders.

    fprintf('Loading training data from: %s\n', datasetPath);
    [X_train, y_train] = loadDataset(fullfile(datasetPath, 'Training'));
    [X_test, y_test]   = loadDataset(fullfile(datasetPath, 'Testing'));
    y_test = setcats(y_test, categories(y_train));

    numFeatures = size(X_train, 2);
    featureMask = true(1, numFeatures);
    X_tr = X_train(:, featureMask);
    X_te = X_test(:, featureMask);

    % Z-score normalisation
    [X_tr_norm, mu_train, std_train] = normalize(X_tr);
    X_te_norm = normalize(X_te, 'center', mu_train, 'scale', std_train);

    fprintf('\n Phase 1/2: Starting Grid Search & CV...\n');
    bestParams = svm_grid_search(X_tr_norm, y_train);
    kernelType = bestParams.kernelType;
    bestC = bestParams.C;
    bestGamma  = bestParams.gamma;

    fprintf('\n Phase 2/2: Training final SVM...\n');
    results = svm_classifier(X_tr_norm, y_train, X_te_norm, y_test, kernelType, bestC, bestGamma);

    model = results.model;
    classNames = results.classNames;
    m = results.metrics;

    fprintf('\n=== Evaluation on Test Set ===\n');
    fprintf('  Accuracy    : %.2f%%\n', m.accuracy * 100);
    fprintf('  Sensitivity : %.2f%%\n', m.sensitivity * 100);
    fprintf('  Specificity : %.2f%%\n', m.specificity * 100);
    fprintf('  Precision   : %.2f%%\n', m.precision * 100);
    fprintf('  F1 Score    : %.2f%%\n', m.f1 * 100);
    fprintf('\nConfusion Matrix:\n');
    disp(m.confMat);
end


function [X, y] = loadDataset(rootDir) 
    imds = imageDatastore(rootDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    numImages = numel(imds.Files);
    
    % Estrai feature della prima immagine per capire la dimensione
    first_feat = extractFeaturesFromFile(imds.Files{1});
    numFeatures = length(first_feat);
    X = zeros(numImages, numFeatures);
    
    fprintf('  Inizio elaborazione...\n', numImages);
    for i = 1:numImages
        X(i, :) = extractFeaturesFromFile(imds.Files{i});
    end
    
    y = imds.Labels;
end

%  Full pipeline for one image file -> feature vector
function features = extractFeaturesFromFile(imgPath)
    img_preprocessed = preprocessing(imgPath);
    [~, brain] = skull_stripping(img_preprocessed);
    [tumor_mask, ~] = tumor_extraction(img_preprocessed, brain);
    features = feature_extraction(tumor_mask, img_preprocessed);
end
