function main(varargin)
% MAIN  Entry point for the Brain Tumour MRI Classification system.
%
%   Usage:
%       main('train')                         % train with default RBF kernel
%       main('train', kernel)                 % e.g. main('train', 'rbf')
%       main('gridsearch')                    % grid search + train, RBF kernel
%       main('gridsearch', kernel)            % e.g. main('gridsearch', 'linear')
%       main('gridsearch', kernel, 'KFolds', 5, 'CValues', logspace(-2,3,6))
%
%   Modes:
%       'train'      - Standard training with the specified kernel (no grid search).
%       'gridsearch' - Runs a k-fold cross-validated grid search over BoxConstraint
%                      and KernelScale, then trains the final model with the best
%                      parameters found.
%
%   Grid Search Name-Value Pairs (only with 'gridsearch' mode):
%       'KFolds'      - Number of CV folds           (default: 5)
%       'CValues'     - BoxConstraint grid values    (default: logspace(-2,3,6))
%       'GammaValues' - KernelScale grid values      (default: logspace(-3,2,6))

    addpath(fileparts(mfilename('fullpath')));

    if nargin == 0
        error('No mode specified. Usage: main(''train'') or main(''gridsearch'')');
    end

    mode = lower(varargin{1});

    switch mode

        % ----------------------------------------------------------
        case 'train'
        % ----------------------------------------------------------
            kernel = 'rbf';
            if nargin >= 2
                kernel = varargin{2};
            end

            datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset');
            train_model(datasetPath, kernel);

        % ----------------------------------------------------------
        case 'gridsearch'
        % ----------------------------------------------------------
            kernel = 'auto'; % Default for grid search tests all 4 kernels
            if nargin >= 2
                kernel = varargin{2};
            end

            % Collect any remaining name-value pairs (KFolds, CValues, GammaValues)
            extraArgs = {};
            if nargin >= 3
                extraArgs = varargin(3:end);
            end

            datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset');
            train_model(datasetPath, kernel, 'GridSearch', true, extraArgs{:});

        % ----------------------------------------------------------
        otherwise
        % ----------------------------------------------------------
            error('Unknown mode: ''%s''. Use ''train'' or ''gridsearch''.', mode);
    end
end
