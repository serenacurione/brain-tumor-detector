clc; clear; close all;

addpath(fileparts(mfilename('fullpath')));
datasetPath = fullfile(fileparts(mfilename('fullpath')), 'dataset');

train_model(datasetPath);
