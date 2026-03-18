function [bestParams, gsResults] = svm_grid_search(X_train, y_train, kernels, varargin)
% SVM_GRID_SEARCH  Exhaustive grid search + k-fold cross-validation for SVM.
%
%   [bestParams, gsResults] = SVM_GRID_SEARCH(X_train, y_train, kernels)
%   [bestParams, gsResults] = SVM_GRID_SEARCH(..., 'KFolds', 5,
%                                 'CValues', [0.1 1 10 100],
%                                 'GammaValues', [0.001 0.01 0.1 1])
%
%   Inputs:
%       X_train    - N x F normalised feature matrix (training set only)
%       y_train    - N x 1 categorical labels
%       kernels    - 'auto' (tests all 4 kernels), a single string like 'rbf', 
%                    or a cell array of strings e.g. {'rbf', 'linear'}
%
%   Name-Value Pairs:
%       'KFolds'       - Number of CV folds            (default: 5)
%       'CValues'      - Grid of BoxConstraint values  (default: logspace(-2,3,6))
%       'GammaValues'  - Grid of KernelScale values    (default: logspace(-3,2,6))
%                        (ignored for the linear kernel)
%
%   Outputs:
%       bestParams - Struct with fields identifying the global best parameter set:
%                    .kernelType - Best kernel string
%                    .C          - Best BoxConstraint
%                    .gamma      - Best KernelScale  (NaN for linear)
%                    .accuracy   - Best CV accuracy
%       gsResults  - Struct array (one per kernel) with the full grid results:
%                    .kernelType - Kernel string evaluated
%                    .C_grid     - BoxConstraint grid tested
%                    .gamma_grid - KernelScale grid tested
%                    .cvAccuracy - accuracy matrix
%                    .bestIdx    - [row col] index into cvAccuracy for this kernel

    % ------------------------------------------------------------------
    % Parse inputs
    % ------------------------------------------------------------------
    p = inputParser();
    p.addRequired('X_train');
    p.addRequired('y_train');
    p.addRequired('kernels');
    p.addParameter('KFolds',      5,                       @(x) isnumeric(x) && x >= 2);
    p.addParameter('CValues',     logspace(-2, 3, 6),      @isnumeric);
    p.addParameter('GammaValues', logspace(-3, 2, 6),      @isnumeric);
    p.parse(X_train, y_train, kernels, varargin{:});

    kFolds     = p.Results.KFolds;
    CValues    = p.Results.CValues;
    GammaValues= p.Results.GammaValues;

    % Resolve kernels array
    if ischar(kernels) || isstring(kernels)
        kernels = {char(kernels)};
    end
    if isscalar(kernels) && strcmpi(strtrim(kernels{1}), 'auto')
        kernels = {'rbf', 'linear', 'polynomial', 'quadratic'};
    end

    % ------------------------------------------------------------------
    % Convert labels to cellstr for fitcecoc
    % ------------------------------------------------------------------
    if iscategorical(y_train)
        y_fit = cellstr(y_train);
    elseif isnumeric(y_train)
        y_fit = cellstr(string(y_train));
    else
        y_fit = y_train;
    end

    % ------------------------------------------------------------------
    % Build k-fold partition (stratified by class)
    % ------------------------------------------------------------------
    cvp = cvpartition(categorical(y_fit), 'KFold', kFolds, 'Stratify', true);

    globalMaxAcc = -1;
    bestParams   = [];

    % Pre-allocate gsResults array
    % We will fill this iteratively
    gsResults = struct('kernelType', {}, 'C_grid', {}, 'gamma_grid', {}, ...
                       'cvAccuracy', {}, 'bestIdx', {});

    % ------------------------------------------------------------------
    % Outer loop: Kernels
    % ------------------------------------------------------------------
    for kIdx = 1:numel(kernels)
        kernelType = lower(strtrim(kernels{kIdx}));

        isLinear = strcmp(kernelType, 'linear');
        
        C_grid = CValues;
        if isLinear
            gamma_grid = NaN;   % only test C for linear
        else
            gamma_grid = GammaValues;
        end

        switch kernelType
            case 'rbf'
                kernelFcn = 'rbf';       polyOrder = [];
            case 'linear'
                kernelFcn = 'linear';    polyOrder = [];
            case {'polynomial', 'quadratic'}
                kernelFcn = 'polynomial'; polyOrder = 2;
            otherwise
                warning('svm_grid_search:unknownKernel', ...
                      'Unknown kernel: ''%s''. Skipping.', kernelType);
                continue;
        end

        nC     = numel(C_grid);
        nGamma = numel(gamma_grid);
        cvAccuracy = zeros(nC, nGamma);

        % ------------------------------------------------------------------
        % Grid search inner loops
        % ------------------------------------------------------------------
        for ci = 1:nC
            C_val = C_grid(ci);

            for gi = 1:nGamma
                gamma_val = gamma_grid(gi);

                foldAcc = zeros(kFolds, 1);

                for fold = 1:kFolds
                    % Split
                    X_cv_train = X_train(training(cvp, fold), :);
                    X_cv_val   = X_train(test(cvp,     fold), :);
                    y_cv_train = y_fit(training(cvp, fold));
                    y_cv_val   = y_fit(test(cvp, fold));

                    % Build learner template
                    if isLinear || isnan(gamma_val)
                        if ~isempty(polyOrder)
                            t = templateSVM('KernelFunction', kernelFcn, ...
                                            'PolynomialOrder', polyOrder, ...
                                            'BoxConstraint', C_val);
                        else
                            t = templateSVM('KernelFunction', kernelFcn, ...
                                            'BoxConstraint', C_val);
                        end
                    else
                        if ~isempty(polyOrder)
                            t = templateSVM('KernelFunction', kernelFcn, ...
                                            'PolynomialOrder', polyOrder, ...
                                            'BoxConstraint', C_val, ...
                                            'KernelScale', gamma_val);
                        else
                            t = templateSVM('KernelFunction', kernelFcn, ...
                                            'BoxConstraint', C_val, ...
                                            'KernelScale', gamma_val);
                        end
                    end

                    % Train fold model
                    mdl = fitcecoc(X_cv_train, y_cv_train, ...
                                   'Learners', t, ...
                                   'Coding',   'onevsall');

                    % Evaluate
                    pred = predict(mdl, X_cv_val);
                    foldAcc(fold) = mean(strcmp(pred, y_cv_val));
                end

                cvAccuracy(ci, gi) = mean(foldAcc);
            end
        end

        % Locate best combo for THIS kernel
        [maxAccKernel, linIdx] = max(cvAccuracy(:));
        [bestCi, bestGi] = ind2sub(size(cvAccuracy), linIdx);

        gsResults(kIdx).kernelType = kernelType;
        gsResults(kIdx).C_grid     = C_grid;
        gsResults(kIdx).gamma_grid = gamma_grid;
        gsResults(kIdx).cvAccuracy = cvAccuracy;
        gsResults(kIdx).bestIdx    = [bestCi, bestGi];

        % Check if this kernel's best beats the global best
        if maxAccKernel > globalMaxAcc
            globalMaxAcc = maxAccKernel;
            bestParams.kernelType = kernelType;
            bestParams.C          = C_grid(bestCi);
            bestParams.gamma      = gamma_grid(bestGi);
            bestParams.accuracy   = maxAccKernel;
        end
    end
end
