function main(datasetPath)
    addpath(fileparts(mfilename('fullpath')));

    if nargin < 1 || isempty(datasetPath)
        datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset');
    end

    train_model(datasetPath);
end
