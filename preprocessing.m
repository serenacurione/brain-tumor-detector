function [filtered_high, filtered_low, filtered_median] = preprocessing(img)
% PREPROCESSING Apply high-pass, low-pass and median filters to a grayscale MRI image.

    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2double(img);

    % High-pass filter (Laplacian sharpening)
    h_high = fspecial('laplacian', 0.2);
    lap = imfilter(img, h_high, 'replicate');
    filtered_high = img - lap;
    filtered_high = mat2gray(filtered_high);

    % Low-pass filter (Gaussian smoothing)
    h_low = fspecial('gaussian', [5 5], 1.5);
    filtered_low = imfilter(img, h_low, 'replicate');
    filtered_low = mat2gray(filtered_low);

    % Median filter (salt-and-pepper / high-freq noise removal)
    filtered_median = medfilt2(img, [3 3]);
    filtered_median = mat2gray(filtered_median);
end
