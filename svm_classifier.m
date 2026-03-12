function results = svm_classifier(X_train, y_train, X_test, y_test, kernelType)
% SVM_CLASSIFIER Train and evaluate a multi-class SVM with a given kernel.
%
%   results = SVM_CLASSIFIER(X_train, y_train, X_test, y_test, kernelType)
%
%   Inputs:
%       X_train    - N_train x F normalised feature matrix
%       y_train    - N_train x 1 categorical label vector
%       X_test     - N_test x F normalised feature matrix
%       y_test     - N_test x 1 categorical label vector
%       kernelType - One of: 'rbf' | 'linear' | 'polynomial' | 'quadratic'
%   Output:
%       results - Struct with fields:
%                 .model       trained ClassificationSVM / ECOC model
%                 .predictions N_test x 1 predictions
%                 .metrics     struct(accuracy, sensitivity, specificity, confMat)
%                 .kernelType  string

    kernelFcn = resolveKernel(kernelType);
    polyOrder = 2;
    if strcmpi(kernelType, 'polynomial') || strcmpi(kernelType, 'quadratic')
        polyOrder = 2;
    end

    templateArgs = buildTemplateArgs(kernelFcn, polyOrder);
    t = templateSVM(templateArgs{:});

    model = fitcecoc(X_train, y_train, 'Learners', t, 'Coding', 'onevsall', 'Standardize', true);

    predictions = predict(model, X_test);

    metrics = evaluate_metrics(y_test, predictions);

    results.model = model;
    results.predictions = predictions;
    results.metrics = metrics;
    results.kernelType = kernelType;
end

% =========================================================================
% Resolve kernel string to MATLAB name
% =========================================================================
function name = resolveKernel(k)
    switch lower(k)
        case 'rbf'
            name = 'rbf';
        case 'linear'
            name = 'linear';
        case {'polynomial', 'quadratic'}
            name = 'polynomial';
        otherwise
            error('Unknown kernel type: %s', k);
    end
end

% =========================================================================
% Build templateSVM argument list
% =========================================================================
function args = buildTemplateArgs(kernelFcn, polyOrder)
    base = {'KernelFunction', kernelFcn, 'KernelScale', 'auto'};
    if strcmp(kernelFcn, 'polynomial')
        args = [base, {'PolynomialOrder', polyOrder}];
    else
        args = base;
    end
end
