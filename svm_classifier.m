function results = svm_classifier(X_train, y_train, X_test, y_test, kernelType)
% SVM_CLASSIFIER  Train and evaluate a multi-class SVM with a given kernel.
%
%   results = SVM_CLASSIFIER(X_train, y_train, X_test, y_test, kernelType)
%
%   Inputs:
%       X_train    - N_train x F normalised feature matrix
%       y_train    - N_train x 1 categorical (or cellstr/numeric) labels
%       X_test     - N_test  x F normalised feature matrix
%       y_test     - N_test  x 1 categorical (or cellstr/numeric) labels
%       kernelType - 'rbf' (default) | 'linear' | 'polynomial' | 'quadratic'
%
%   Output:
%       results.model        - Trained fitcecoc model
%       results.classNames   - Class name strings
%       results.predictions  - N_test x 1 categorical predictions
%       results.metrics      - struct from evaluate_metrics
%       results.kernelType   - kernel string used

    if nargin < 5 || isempty(kernelType)
        kernelType = 'rbf';
    end

    % ------------------------------------------------------------------
    % Map user-friendly kernel name -> MATLAB KernelFunction string
    % ------------------------------------------------------------------
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
        otherwise
            error('svm_classifier:unknownKernel', ...
                  'Unknown kernel: ''%s''. Use rbf|linear|polynomial|quadratic.', kernelType);
    end

    % ------------------------------------------------------------------
    % fitcecoc does NOT accept categorical Y — convert to cellstr.
    % This preserves the class names and ordering.
    % ------------------------------------------------------------------
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

    % ------------------------------------------------------------------
    % Build SVM learner via templateSVM() — required by fitcecoc.
    % A plain struct is NOT accepted and causes the "cell" type error.
    % ------------------------------------------------------------------
    if ~isempty(polyOrder)
        learner = templateSVM('KernelFunction', kernelFcn, ...
                              'PolynomialOrder', polyOrder);
    else
        learner = templateSVM('KernelFunction', kernelFcn);
    end

    % ------------------------------------------------------------------
    % Train multi-class ECOC model (cellstr labels, templateSVM learner)
    % ------------------------------------------------------------------
    fprintf('[svm_classifier] Fitting ECOC model (%s kernel)...\n', kernelType);
    model = fitcecoc(X_train, y_train_fit, ...
        'Learners', learner, ...
        'Coding',   'onevsall');

    % ------------------------------------------------------------------
    % Predict and convert back to categorical
    % ------------------------------------------------------------------
    pred_raw    = predict(model, X_test);
    if iscell(pred_raw)
        predictions = categorical(pred_raw, classNames);
    else
        predictions = categorical(cellstr(string(pred_raw)), classNames);
    end

    y_test_cat = categorical(y_test_fit, classNames);

    % ------------------------------------------------------------------
    % Evaluate
    % ------------------------------------------------------------------
    metrics = evaluate_metrics(y_test_cat, predictions);

    results.model       = model;
    results.classNames  = classNames;
    results.predictions = predictions;
    results.metrics     = metrics;
    results.kernelType  = kernelType;
end
