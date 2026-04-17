clc; clear; close all;

addpath(fileparts(mfilename('fullpath')));
datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset');

% Cartella di output dove salvare le immagini risultanti
outDir = fullfile(fileparts(mfilename('fullpath')), 'output_segmentations');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fprintf('Loading data from: %s\n', datasetPath);

% Utilizziamo imageDatastore come in loadDataset per attraversare tutto il dataset
imds = imageDatastore(datasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Raccogliamo un massimo di 5 immagini per ogni categoria
filesToProcess = {};
cats = categories(imds.Labels);
for c = 1:length(cats)
    idx = find(imds.Labels == cats{c});
    numToGet = min(5, length(idx));
    
    % Aggiungiamo alla lista di elaborazione le prime 5 trovate per questa classe
    % (si può applicare un randperm se le si vogliono casuali)
    filesToProcess = [filesToProcess; imds.Files(idx(1:numToGet))];
end

numImagesToProcess = length(filesToProcess);
fprintf('Inizio elaborazione e salvataggio (%d immagini totali, 5 per classe)...\n', numImagesToProcess);

for i = 1:numImagesToProcess
    imgPath = filesToProcess{i};
    [parentDir, fname, ext] = fileparts(imgPath);
    
    % Estraiamo il nome della categoria dalla cartella
    [~, catName] = fileparts(parentDir);
    
    % 1. Preprocessing (che include caricamento e resize)
    img_preprocessed = preprocessing(imgPath);

    % 2. Skull Stripping
    [skull_mask, brain] = skull_stripping(img_preprocessed);
    
    % 3. Tumor Extraction
    [tumor_mask, tumor] = tumor_extraction(img_preprocessed, brain);
    
    % 4. Salvataggio Immagini
    baseName = sprintf('%s_%s', catName, fname);
    
    % Crea una cartella specifica per questa immagine
    imageDir = fullfile(outDir, sprintf('image_%d', i));
    if ~exist(imageDir, 'dir')
        mkdir(imageDir);
    end
    
    % a) Originale Processata (Base per i confronti)
    imwrite(img_preprocessed, fullfile(imageDir, sprintf('%s_0_preprocessed%s', baseName, ext)));
    
    % b) Skull Stripping Mask
    imwrite(skull_mask, fullfile(imageDir, sprintf('%s_1_skull_mask%s', baseName, ext)));
    
    % c) Brain (Extracted)
    imwrite(brain, fullfile(imageDir, sprintf('%s_2_brain%s', baseName, ext)));
    
    % d) Tumor Mask
    imwrite(tumor_mask, fullfile(imageDir, sprintf('%s_3_tumor_mask%s', baseName, ext)));
    
    % e) Tumor
    imwrite(tumor, fullfile(imageDir, sprintf('%s_4_tumor%s', baseName, ext)));
end

fprintf('Tutte le immagini sono state salvate con successo nella cartella "%s"!\n', outDir);
