function bestParams = svm_grid_search(X_train, y_train)
%   Inputs:
%       X_train    - N x F normalised feature matrix (training set only)
%       y_train    - N x 1 categorical labels
%
%   Outputs:
%       bestParams - Struct with fields identifying the global best parameter set:
%                    .kernelType - Best kernel string
%                    .C          - Best BoxConstraint
%                    .gamma      - Best KernelScale  (NaN for linear)
%                    .accuracy   - Best CV accuracy

    % Grid search configuration parameters
    kFolds      = 5;
    CValues     = logspace(-2, 3, 6);
    GammaValues = logspace(-3, 2, 6);
    kernels     = {'rbf', 'linear', 'polynomial', 'quadratic'};

    % Convert labels to cellstr for fitcecoc
    if iscategorical(y_train)
        y_fit = cellstr(y_train);
    elseif isnumeric(y_train)
        y_fit = cellstr(string(y_train));
    else
        y_fit = y_train;
    end

    % Build k-fold partition (stratified by class)
    cvp = cvpartition(categorical(y_fit), 'KFold', kFolds, 'Stratify', true);
    cvData = cell(kFolds, 1);
    for fold = 1:kFolds
        cvData{fold}.X_train = X_train(training(cvp, fold), :);
        cvData{fold}.X_val   = X_train(test(cvp, fold), :);
        cvData{fold}.y_train = y_fit(training(cvp, fold));
        cvData{fold}.y_val   = y_fit(test(cvp, fold));
    end

    globalMaxAcc = -1;
    bestParams   = [];

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
                kernelFcn = 'rbf';       
                polyOrder = [];
            case 'linear'
                kernelFcn = 'linear';    
                polyOrder = [];
            case {'polynomial', 'quadratic'}
                kernelFcn = 'polynomial'; 
                polyOrder = 2;
            otherwise
                continue;
        end

        nC = numel(C_grid);
        nGamma = numel(gamma_grid);
        cvAccuracy = zeros(nC, nGamma);

        for ci = 1:nC
            C_val = C_grid(ci);

            for gi = 1:nGamma
                gamma_val = gamma_grid(gi);

                svmArgs = {'KernelFunction', kernelFcn, 'BoxConstraint', C_val};
                if ~isempty(polyOrder)
                    svmArgs = [svmArgs, {'PolynomialOrder', polyOrder}];
                end
                if ~isLinear && ~isnan(gamma_val)
                    svmArgs = [svmArgs, {'KernelScale', gamma_val}];
                end
                t = templateSVM(svmArgs{:});
                foldAcc = zeros(kFolds, 1);

                for fold = 1:kFolds
                    mdl = fitcecoc(cvData{fold}.X_train, cvData{fold}.y_train, ...
                                   'Learners', t, ...
                                   'Coding',   'onevsall');

                    pred = predict(mdl, cvData{fold}.X_val);
                    foldAcc(fold) = mean(strcmp(pred, cvData{fold}.y_val));
                end

                cvAccuracy(ci, gi) = mean(foldAcc);
            end
        end

        % Locate best combo for THIS kernel
        [maxAccKernel, linIdx] = max(cvAccuracy(:));
        [bestCi, bestGi] = ind2sub(size(cvAccuracy), linIdx);

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
