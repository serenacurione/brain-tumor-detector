function main(varargin)
% MAIN  Entry point for the Brain Tumour MRI Classification system.
%
%   Usage:
%       main()                     % opens the GUI
%       main('train')              % trains the model from CLI (no GUI)
%       main('train', kernel)      % e.g. main('train','rbf')
%       main('train', kernel, 1)   % use GA for feature selection

    addpath(fileparts(mfilename('fullpath')));

    if nargin == 0
        BrainTumorGUI();
        return;
    end

    mode = lower(varargin{1});
    
    switch mode
        case 'train'
            kernel = 'rbf';
            useGA = false;
            
            if nargin >= 2
                kernel = varargin{2};
            end
            if nargin >= 3
                useGA = logical(varargin{3});
            end

            datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset');
            train_model(datasetPath, kernel, useGA);

        otherwise
            error('Unknown mode: %s. Use ''train'' or call main() with no args.', mode);
    end
end
