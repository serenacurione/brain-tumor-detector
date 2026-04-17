function results = svm_classifier(X_train, y_train, X_test, y_test, kernelType, C_val, gamma_val)
%   Inputs:
%       C_val       - SVM regularisation parameter C (default: 1)
%       gamma_val   - RBF/poly kernel bandwidth gamma (default: NaN)
%
%   Output:
%       results.model        - Trained fitcecoc model
%       results.classNames   - Class name strings
%       results.predictions  - N_test x 1 categorical predictions
%       results.metrics      - struct from evaluate_metrics
%       results.kernelType   - kernel string used
%       results.C            - BoxConstraint used
%       results.gamma        - KernelScale used (NaN for linear)

    % Map user-friendly kernel name -> MATLAB KernelFunction string
    switch lower(strtrim(kernelType))
        case 'rbf'
            kernelFcn = 'rbf';
            polyOrder = [];
        case 'linear'
            kernelFcn = 'linear';
            polyOrder = [];
        case {'polynomial', 'quadratic'}
            kernelFcn = 'polynomial';
            polyOrder = 2;
    end

    if iscategorical(y_train)
        classNames   = categories(y_train);
        y_train_fit  = cellstr(y_train);
        y_test_fit   = cellstr(y_test);
    elseif isnumeric(y_train)
        classNames   = cellstr(string(unique(y_train)));
        y_train_fit  = cellstr(string(y_train));
        y_test_fit   = cellstr(string(y_test));
    else
        % already cellstr
        classNames   = unique(y_train);
        y_train_fit  = y_train;
        y_test_fit   = y_test;
    end

    % KernelScale is only meaningful for rbf/polynomial kernels.
    useGamma = ~isnan(gamma_val) && ~strcmp(kernelFcn, 'linear');

    svmArgs = {'KernelFunction', kernelFcn, 'BoxConstraint', C_val};
    if ~isempty(polyOrder)
        svmArgs = [svmArgs, {'PolynomialOrder', polyOrder}];
    end
    if useGamma
        svmArgs = [svmArgs, {'KernelScale', gamma_val}];
    end
    learner = templateSVM(svmArgs{:});

    model = fitcecoc(X_train, y_train_fit, ...
        'Learners', learner, ...
        'Coding',   'onevsall');

    pred_raw    = predict(model, X_test);
    if iscell(pred_raw)
        predictions = categorical(pred_raw, classNames);
    else
        predictions = categorical(cellstr(string(pred_raw)), classNames);
    end

    y_test_cat = categorical(y_test_fit, classNames);
    metrics = evaluate_metrics(y_test_cat, predictions);

    results.model       = model;
    results.classNames  = classNames;
    results.predictions = predictions;
    results.metrics     = metrics;
    results.kernelType  = kernelType;
    results.C           = C_val;
    results.gamma       = gamma_val;
end
