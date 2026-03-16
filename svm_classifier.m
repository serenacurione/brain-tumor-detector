function results = svm_classifier(X_train, y_train, X_test, y_test, kernelType, varargin)
% SVM_CLASSIFIER  Train and evaluate a multi-class SVM with a given kernel.
%
%   results = SVM_CLASSIFIER(X_train, y_train, X_test, y_test, kernelType)
%   results = SVM_CLASSIFIER(..., 'BoxConstraint', C, 'KernelScale', gamma)
%
%   Inputs:
%       X_train    - N_train x F normalised feature matrix
%       y_train    - N_train x 1 categorical (or cellstr/numeric) labels
%       X_test     - N_test  x F normalised feature matrix
%       y_test     - N_test  x 1 categorical (or cellstr/numeric) labels
%       kernelType - 'rbf' (default) | 'linear' | 'polynomial' | 'quadratic'
%
%   Optional Name-Value Pairs:
%       'BoxConstraint' - SVM regularisation parameter C (default: 1)
%       'KernelScale'   - RBF/poly kernel bandwidth gamma (default: 'auto')
%
%   Output:
%       results.model        - Trained fitcecoc model
%       results.classNames   - Class name strings
%       results.predictions  - N_test x 1 categorical predictions
%       results.metrics      - struct from evaluate_metrics
%       results.kernelType   - kernel string used
%       results.C            - BoxConstraint used
%       results.gamma        - KernelScale used (NaN for linear)

    if nargin < 5 || isempty(kernelType)
        kernelType = 'rbf';
    end

    % ------------------------------------------------------------------
    % Parse optional hyperparameter overrides (from grid search)
    % ------------------------------------------------------------------
    p = inputParser();
    p.addParameter('BoxConstraint', 1,    @(x) isnumeric(x) && x > 0);
    p.addParameter('KernelScale',   NaN,  @(x) isnumeric(x));
    p.parse(varargin{:});

    C_val     = p.Results.BoxConstraint;
    gamma_val = p.Results.KernelScale;

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
    % Build SVM learner via templateSVM() with hyperparameters.
    % KernelScale is only meaningful for rbf/polynomial kernels.
    % ------------------------------------------------------------------
    useGamma = ~isnan(gamma_val) && ~strcmp(kernelFcn, 'linear');

    if ~isempty(polyOrder)
        if useGamma
            learner = templateSVM('KernelFunction',  kernelFcn, ...
                                  'PolynomialOrder', polyOrder, ...
                                  'BoxConstraint',   C_val, ...
                                  'KernelScale',     gamma_val);
        else
            learner = templateSVM('KernelFunction',  kernelFcn, ...
                                  'PolynomialOrder', polyOrder, ...
                                  'BoxConstraint',   C_val);
        end
    else
        if useGamma
            learner = templateSVM('KernelFunction', kernelFcn, ...
                                  'BoxConstraint',  C_val, ...
                                  'KernelScale',    gamma_val);
        else
            learner = templateSVM('KernelFunction', kernelFcn, ...
                                  'BoxConstraint',  C_val);
        end
    end

    % ------------------------------------------------------------------
    % Train multi-class ECOC model (cellstr labels, templateSVM learner)
    % ------------------------------------------------------------------
    if useGamma
        fprintf('[svm_classifier] Fitting ECOC model (%s kernel | C=%.4g | gamma=%.4g)...\n', ...
            kernelType, C_val, gamma_val);
    else
        fprintf('[svm_classifier] Fitting ECOC model (%s kernel | C=%.4g)...\n', ...
            kernelType, C_val);
    end

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
    results.C           = C_val;
    results.gamma       = gamma_val;
end
