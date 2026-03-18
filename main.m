function main(datasetPath)
    % MAIN  Entry point for the Brain Tumour MRI Classification system.
    %
    %   Usage:
    %       main()                 % Grid search + train, with default 'dataset' folder
    %       main('custom/path')    % Grid search + train, with custom dataset path
    %
    %   This function always performs a k-fold cross-validated grid search 
    %   comparing all SVM kernels to find the best hyperparameters.

    addpath(fileparts(mfilename('fullpath')));

    if nargin < 1 || isempty(datasetPath)
        datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset');
    end

    % Always runs a comprehensive grid search
    train_model(datasetPath);
end
