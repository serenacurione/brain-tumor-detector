function selected = genetic_feature_selection(X, y, numFeatures)
% GENETIC_FEATURE_SELECTION Select the best feature subset via a Genetic Algorithm.
%
%   selected = GENETIC_FEATURE_SELECTION(X, y, numFeatures)
%
%   Inputs:
%       X - N x F feature matrix (N samples, F features)
%       y - N x 1 label vector
%       numFeatures - Total number of features (F)
%   Output:
%       selected - Logical 1 x F mask; true = feature is selected

    opts = optimoptions('ga', 'PopulationSize', 50, 'MaxGenerations', 30, 'CrossoverFraction', 0.8, 'MutationFcn', @mutationuniform, 'Display', 'off', 'UseParallel', false);

    fitnessFcn = @(mask) gaFitness(mask, X, y);

    nvars = numFeatures;
    lb = zeros(1, nvars);
    ub = ones(1, nvars);
    intcon = 1:nvars;

    bestMask = ga(fitnessFcn, nvars, [], [], [], [], lb, ub, [], intcon, opts);

    selected = logical(bestMask);

    % guarantee at least one feature is chosen
    if ~any(selected)
        selected(1) = true;
    end
end

% =========================================================================
% Fitness function: negative 5-fold cross-validated SVM accuracy
% =========================================================================
function score = gaFitness(mask, X, y)
    mask = logical(mask);
    if ~any(mask)
        score = 1;
        return;
    end
    Xsub = X(:, mask);
    try
        cv = cvpartition(y, 'KFold', 5, 'Stratify', true);
        acc = zeros(cv.NumTestSets, 1);
        for k = 1:cv.NumTestSets
            Xtr = Xsub(cv.training(k), :);
            ytr = y(cv.training(k));
            Xte = Xsub(cv.test(k), :);
            yte = y(cv.test(k));
            mdl = fitcsvm(Xtr, ytr, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'Standardize', true);
            pred = predict(mdl, Xte);
            acc(k) = mean(pred == yte);
        end
        score = -mean(acc); % GA minimises, so negate accuracy
    catch
        score = 1;
    end
end
