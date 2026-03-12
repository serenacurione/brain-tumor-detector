function BrainTumorGUI()
% BRAINTUMORGUI  Graphical User Interface for Brain Tumour Classification.
%
%   Launch with:
%       BrainTumorGUI()
%   or via main.m.
%
%   The GUI guides the user through:
%       1. Load MRI image
%       2. Pre-processing  (high-pass / low-pass / median filters)
%       3. Segmentation    (EM, Level-Set, Watershed, Threshold)
%       4. Feature extraction
%       5. SVM classification
%       6. Results display (original, filtered, bounding-box, tumour)

    % ------------------------------------------------------------------ %
    %  Shared state
    % ------------------------------------------------------------------ %
    state = struct( ...
        'imgOriginal',    [], ...
        'imgGray',        [], ...
        'imgFiltered',    [], ...
        'segMask',        [], ...
        'features',       [], ...
        'prediction',     '', ...
        'model',          [], ...
        'classNames',     [], ...
        'featureMask',    [], ...
        'mu_train',       [], ...
        'std_train',      [], ...
        'classMap',       [], ...
        'datasetPath',    '' ...
    );

    % ------------------------------------------------------------------ %
    % Colour palette %
    % ------------------------------------------------------------------ %
    BG_DARK      = [0.08 0.09 0.12];
    BG_PANEL     = [0.11 0.13 0.17];
    BG_CARD      = [0.15 0.17 0.22];
    ACCENT_BLUE  = [0.20 0.55 0.95];
    ACCENT_GREEN = [0.18 0.78 0.56];
    ACCENT_RED   = [0.95 0.33 0.33];
    TEXT_MAIN    = [0.95 0.95 0.97];
    TEXT_SUB     = [0.60 0.63 0.72];

    % ------------------------------------------------------------------ %
    % Main window %
    % ------------------------------------------------------------------ %
    fig = uifigure('Name', 'Brain Tumor MRI Classifier', 'Position', [50 50 1400 820], 'Color', BG_DARK, 'Resize', 'on', 'AutoResizeChildren', 'off');

    % ------------------------------------------------------------------ %
    % Left control panel %
    % ------------------------------------------------------------------ %
    ctrlPanel = uipanel(fig, 'Position', [10 10 260 800], 'BackgroundColor', BG_PANEL, 'BorderType', 'none');

    % Title 
    uilabel(ctrlPanel, 'Text', 'Brain Tumor', 'Position', [10 755 240 28], 'FontSize', 20, 'FontWeight', 'bold', 'FontColor', TEXT_MAIN, 'HorizontalAlignment', 'center');
    uilabel(ctrlPanel, 'Text', 'MRI Classifier', 'Position', [10 730 240 22], 'FontSize', 13, 'FontColor', ACCENT_BLUE, 'HorizontalAlignment', 'center');

    uipanel(ctrlPanel, 'Position', [20 720 220 2], 'BackgroundColor', BG_CARD, 'BorderType', 'none');

    % --Buttons-- % 
    btnW = 220;
    btnH = 38;
    btnX = 20;
    btnY = 680;
    GAP = 12;

    btnLoad = makeBtn(ctrlPanel, 'Load MRI Image', [btnX btnY btnW btnH], ACCENT_BLUE, TEXT_MAIN, @cb_loadImage);
    btnY = btnY - btnH - GAP;

    % Dataset path field 
    uilabel(ctrlPanel, 'Text', 'Dataset path:', 'Position', [btnX btnY+btnH+4 btnW 16], 'FontSize', 10, 'FontColor', TEXT_SUB);
    edtDataset = uieditfield(ctrlPanel, 'text', 'Value', '', 'Position', [btnX btnY btnW 30], 'BackgroundColor', BG_CARD, 'FontColor', TEXT_MAIN, 'FontSize', 10);
    btnY = btnY - 30 - GAP;

    % Kernel selector 
    uilabel(ctrlPanel, 'Text', 'SVM Kernel:', 'Position', [btnX btnY+30 btnW 16], 'FontSize', 10, 'FontColor', TEXT_SUB);
    ddKernel = uidropdown(ctrlPanel, 'Items', {'rbf', 'linear', 'polynomial', 'quadratic'}, 'Value', 'rbf', 'Position', [btnX btnY btnW 28], 'BackgroundColor', BG_CARD, 'FontColor', TEXT_MAIN);
    btnY = btnY - 28 - GAP;

    % GA toggle 
    cbGA = uicheckbox(ctrlPanel, 'Text', 'Use Genetic Algorithm', 'Value', false, 'Position', [btnX btnY btnW 24], 'FontColor', TEXT_SUB, 'FontSize', 10);
    btnY = btnY - 24 - GAP;

    btnTrain = makeBtn(ctrlPanel, 'Train Model', [btnX btnY btnW btnH], BG_CARD, ACCENT_BLUE, @cb_trainModel);
    btnY = btnY - btnH - GAP;

    uipanel(ctrlPanel, 'Position', [20 btnY+2 220 2], 'BackgroundColor', BG_CARD, 'BorderType', 'none');
    btnY = btnY - 10;

    btnProcess = makeBtn(ctrlPanel, 'Pre-process', [btnX btnY btnW btnH], BG_CARD, ACCENT_GREEN, @cb_preprocess);
    btnY = btnY - btnH - GAP;

    btnSegment = makeBtn(ctrlPanel, 'Segment', [btnX btnY btnW btnH], BG_CARD, ACCENT_GREEN, @cb_segment);
    btnY = btnY - btnH - GAP;

    btnExtract = makeBtn(ctrlPanel, 'Extract Features', [btnX btnY btnW btnH], BG_CARD, ACCENT_GREEN, @cb_extractFeatures);
    btnY = btnY - btnH - GAP;

    btnClassify = makeBtn(ctrlPanel, 'Classify', [btnX btnY btnW btnH], BG_CARD, ACCENT_GREEN, @cb_classify);
    btnY = btnY - btnH - GAP - 5;

    btnRunAll = makeBtn(ctrlPanel, 'Run Full Pipeline', [btnX btnY btnW btnH+4], ACCENT_BLUE, TEXT_MAIN, @cb_runAll);
    btnRunAll.FontWeight = 'bold';
    btnY = btnY - btnH - GAP * 2;

    uipanel(ctrlPanel, 'Position', [20 btnY+2 220 2], 'BackgroundColor', BG_CARD, 'BorderType', 'none');
    btnY = btnY - 10;

    % Metrics button 
    btnMetrics = makeBtn(ctrlPanel, 'Show Metrics', [btnX btnY btnW btnH], BG_CARD, TEXT_SUB, @cb_showMetrics);
    btnY = btnY - btnH - GAP;
    % #ok<NASGU>

    % Status label 
    lblStatus = uilabel(ctrlPanel, 'Text', 'Ready.', 'Position', [10 10 240 60], 'WordWrap', 'on', 'FontSize', 10, 'FontColor', TEXT_SUB, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top');

    % ------------------------------------------------------------------ %
    % Image display area(4 panels) %
    % ------------------------------------------------------------------ %
    dispX = 280;
    dispY = 10;
    dispW = 1110;
    dispH = 800;
    panelDisp = uipanel(fig, 'Position', [dispX dispY dispW dispH], 'BackgroundColor', BG_DARK, 'BorderType', 'none');

    % Header row 
    uilabel(panelDisp, 'Text', 'Original', 'Position', [5 770 260 22], 'FontColor', TEXT_SUB, 'FontSize', 11, 'HorizontalAlignment', 'center');
    uilabel(panelDisp, 'Text', 'Filtered (Median)', 'Position', [280 770 260 22], 'FontColor', TEXT_SUB, 'FontSize', 11, 'HorizontalAlignment', 'center');
    uilabel(panelDisp, 'Text', 'Segmentation + Bounding Box', 'Position', [555 770 280 22], 'FontColor', TEXT_SUB, 'FontSize', 11, 'HorizontalAlignment', 'center');
    uilabel(panelDisp, 'Text', 'Detected Tumour', 'Position', [845 770 255 22], 'FontColor', TEXT_SUB, 'FontSize', 11, 'HorizontalAlignment', 'center');

    axOrig = uiaxes(panelDisp, 'Position', [5 10 260 755], 'Color', BG_CARD, 'XColor', BG_CARD, 'YColor', BG_CARD);
    axFilt = uiaxes(panelDisp, 'Position', [280 10 260 755], 'Color', BG_CARD, 'XColor', BG_CARD, 'YColor', BG_CARD);
    axSeg = uiaxes(panelDisp, 'Position', [555 10 280 755], 'Color', BG_CARD, 'XColor', BG_CARD, 'YColor', BG_CARD);
    axTumr = uiaxes(panelDisp, 'Position', [845 10 255 755], 'Color', BG_CARD, 'XColor', BG_CARD, 'YColor', BG_CARD);

    styleAxis(axOrig);
    styleAxis(axFilt);
    styleAxis(axSeg);
    styleAxis(axTumr);

    % Result label(overlaid on tumour panel) 
    lblResult = uilabel(panelDisp, 'Text', '', 'Position', [845 720 255 36], 'FontSize', 16, 'FontWeight', 'bold', 'FontColor', ACCENT_GREEN, 'HorizontalAlignment', 'center', 'BackgroundColor', [0 0 0 0]);

    % ------------------------------------------------------------------ %
    % CALLBACKS %
    % ------------------------------------------------------------------ %

    function cb_loadImage(~, ~)
        [fname, fpath] = uigetfile({'*.jpg;*.jpeg;*.png', 'MRI Images'});
        if isequal(fname, 0)
            return;
        end
        img = imread(fullfile(fpath, fname));
        state.imgOriginal = img;
        if size(img, 3) == 3 
            state.imgGray = rgb2gray(img);
        else
            state.imgGray = img;
        end 
        state.imgGray = imresize(state.imgGray, [256 256]);
        imshow(state.imgOriginal, 'Parent', axOrig);
        axOrig.Title.String = '';
        axOrig.Title.Color = TEXT_MAIN;
        clearAxes(axFilt);
        clearAxes(axSeg);
        clearAxes(axTumr);
        lblResult.Text = '';
        state.prediction = '';
        setStatus('Image loaded. Run Pre-process next.');
    end

    function cb_preprocess(~, ~) 
        if isempty(state.imgGray)
            setStatus('Load an image first.');
            return; 
        end 
        setStatus('Pre-processing...');
        [ ~, ~, img_med ] = preprocessing(state.imgGray);
        state.imgFiltered = img_med;
        imshow(img_med, 'Parent', axFilt);
        setStatus('Pre-processing done. Run Segment next.');
    end

    function cb_segment(~, ~) 
        if isempty(state.imgFiltered)
            setStatus('Run Pre-process first.');
            return; 
        end 
        setStatus('Segmenting...');
        [ seg_em, ~, ~, ~] = segmentation(state.imgFiltered);
        state.segMask = seg_em;
        % Overlay on original 
        overlayImg = imoverlay(state.imgFiltered, bwperim(seg_em), [0.2 0.9 0.5]);
        imshow(overlayImg, 'Parent', axSeg);
        drawBoundingBox(axSeg, seg_em, ACCENT_GREEN);
        setStatus('Segmentation done. Extract Features next.');
    end

    function cb_extractFeatures(~, ~) 
        if isempty(state.segMask)
            setStatus('Run Segment first.');
            return;
        end 
        setStatus('Extracting features...');
        feat = feature_extraction(state.segMask, state.imgFiltered);
        state.features = feat;
        setStatus(sprintf('Features extracted (11 values). Classify next.'));
    end

    function cb_classify(~, ~) 
        if isempty(state.features)
            setStatus('Extract features first.');
            return; 
        end 
        if isempty(state.model)
            setStatus('No trained model found. Train the model or load a dataset.');
            return; 
        end 
        setStatus('Classifying...');
        x = state.features(state.featureMask);
        x = (x - state.mu_train) ./ state.std_train;
        pred = predict(state.model, x);
        if iscell(pred)
            state.prediction = pred{1};
        else
            state.prediction = char(pred);
        end

        showTumourPanel(axTumr, state.imgGray, state.segMask, state.prediction, ACCENT_RED, ACCENT_GREEN, TEXT_MAIN);

        resultColor = ACCENT_GREEN;
        if ~strcmpi(state.prediction, 'no_tumor') 
            resultColor = ACCENT_RED;
        end 
        lblResult.FontColor = resultColor;
        lblResult.Text = formatLabel(state.prediction);
        setStatus(['Classification: ' formatLabel(state.prediction)]);
    end

    function cb_trainModel(~, ~) 
        dPath = strtrim(edtDataset.Value);
        if isempty(dPath)
            dPath = uigetdir('', 'Select dataset root folder');
            if isequal(dPath, 0)
                return;
            end 
            edtDataset.Value = dPath;
        end 
        if ~isfolder(dPath) 
            setStatus('Dataset path not found.');
            return;
        end 
        kernel = ddKernel.Value;
        useGA = cbGA.Value;
        state.datasetPath = dPath;

        setStatus(sprintf('Training (%s kernel)... this may take minutes.', kernel));
        drawnow;
        try
            [mdl, fmask, mu, sd] = train_model(dPath, kernel, useGA);
            state.model = mdl;
            state.featureMask = fmask;
            state.mu_train = mu;
            state.std_train = sd;
            saved = load(fullfile(dPath, 'brain_tumor_model.mat'), 'classMap', 'classNames');
            state.classMap   = saved.classMap;
            state.classNames = saved.classNames;
            setStatus('Model trained and saved. Load an image to classify.');
        catch ME 
            setStatus(['Training error: ' ME.message]);
        end 
    end

    function cb_runAll(~, ~) 
        cb_preprocess([], []);
        if isempty(state.imgFiltered)
            return;
        end 
        cb_segment([], []);
        cb_extractFeatures([], []);
        cb_classify([], []);
    end

    function cb_showMetrics(~, ~) 
        dPath = strtrim(edtDataset.Value);
        if isempty(dPath) || isempty(state.model) 
            setStatus('Train the model first.');
            return; 
        end 
        setStatus('Computing metrics on test set...');
        drawnow;
        try 
            kernel = ddKernel.Value;
            [ mdl, fm, mu, sd ] = train_model(dPath, kernel, false);
            state.model = mdl;
            state.featureMask = fm;
            state.mu_train = mu;
            state.std_train = sd;
            setStatus('Metrics displayed in Command Window.');
        catch ME 
            setStatus(['Metrics error: ' ME.message]);
        end 
    end

    % ------------------------------------------------------------------ %
    % Helpers %
    % ------------------------------------------------------------------ %
    function setStatus(msg) 
        lblStatus.Text = msg;
        drawnow;
    end

    % ------------------------------------------------------------------ %
    % Helper : create styled push - button %
    % ------------------------------------------------------------------ %
    function btn = makeBtn(parent, label, pos, bgColor, fgColor, cb) 
        btn = uibutton(parent, 'push', 'Text', label, 'Position', pos, 'BackgroundColor', bgColor, 'FontColor', fgColor, 'FontSize', 12, 'ButtonPushedFcn', cb);
    end

    % ------------------------------------------------------------------ %
    % Style axes(dark background, no ticks) %
    % ------------------------------------------------------------------ %
    function styleAxis(ax) 
        ax.XTick = [];
        ax.YTick = [];
        ax.Box = 'off';
        ax.Color = BG_CARD;
        ax.XColor = BG_CARD;
        ax.YColor = BG_CARD;
    end

    function clearAxes(ax) 
        cla(ax);
        ax.Color = BG_CARD;
    end

    % ------------------------------------------------------------------ %
    % Draw bounding box around largest connected component %
    % ------------------------------------------------------------------ %
    function drawBoundingBox(ax, mask, color) 
        props = regionprops(mask, 'BoundingBox', 'Area');
        if isempty(props)
            return;
        end
        [~, idx] = max([props.Area]);
        bb = props(idx).BoundingBox;
        hold(ax, 'on');
        rectangle(ax, 'Position', bb, 'EdgeColor', color, 'LineWidth', 2.5, 'LineStyle', '--');
        hold(ax, 'off');
    end

    % ------------------------------------------------------------------ %
    % Show tumour crop panel %
    % ------------------------------------------------------------------ %
    function showTumourPanel(ax, imgGray, mask, label, colTumor, colNorm, colText) 
        cla(ax);
        props = regionprops(mask, 'BoundingBox', 'Area');
        if isempty(props) || strcmpi(label, 'no_tumor') 
            imshow(imgGray, 'Parent', ax);
            text(ax, size(imgGray, 2) / 2, size(imgGray, 1) / 2, 'No Tumor', 'Color', colNorm, 'FontSize', 18, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
            return;
        end
        [~, idx] = max([props.Area]);
        bb = round(props(idx).BoundingBox);
        bb(1) = max(1, bb(1));
        bb(2) = max(1, bb(2));
        bb(3) = min(size(imgGray, 2) - bb(1) + 1, bb(3));
        bb(4) = min(size(imgGray, 1) - bb(2) + 1, bb(4));
        if bb(3) < 1 || bb(4) < 1 
            imshow(imgGray, 'Parent', ax);
            return;
        end 
        crop = imcrop(imgGray, bb);
        imshow(crop, 'Parent', ax);
        hold(ax, 'on');
        rectangle(ax, 'Position', [1 1 size(crop, 2)-1 size(crop, 1)-1], 'EdgeColor', colTumor, 'LineWidth', 3);
        text(ax, size(crop, 2) / 2, 10, label, 'Color', colText, 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
        hold(ax, 'off');
    end

    % ------------------------------------------------------------------ %
    %  Format class label for display
    % ------------------------------------------------------------------ %
    function s = formatLabel(raw)
        switch lower(raw)
            case 'no_tumor',         s = 'No Tumor ✓';
            case 'glioma_tumor',     s = 'Glioma Tumor ⚠';
            case 'meningioma_tumor', s = 'Meningioma ⚠';
            case 'pituitary_tumor',  s = 'Pituitary Tumor ⚠';
            otherwise,               s = raw;
        end
    end

    % Suppress unused-variable warnings for button handles
    %#ok<*NASGU>
end
