function [binary_mask, brain] = skull_stripping(img, varargin)

threshold = 30;
binary_img = img > threshold;

% morphological operations
binary_cleaned = bwareaopen(binary_img, 100);
binary_filled = imfill(binary_cleaned, "holes");
se = strel('disk', 40);
binary_mask = imerode(binary_filled, se);

% brain extraction
brain = double(img) .* double(binary_mask);
brain = uint8(brain);

end