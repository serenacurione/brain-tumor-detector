function output_img = preprocessing(input_data)
    % PREPROCESSING Load (if necessary), resize, and apply median, low-pass, and high-pass filters to a grayscale MRI image.

    % If input is a string or character vector (file path), load the image
    if ischar(input_data) || isstring(input_data)
        img = imread(input_data);
    else
        img = input_data;
    end

    % Convert to grayscale if it is an RGB image
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    img = imresize(img, [256 256]);
    img = im2double(img);

    % 1. Median filter (salt-and-pepper / high-freq noise removal)
    filtered_median = medfilt2(img, [3 3]);

    % 2. Low-pass filter (Gaussian smoothing)
    h_low = fspecial('gaussian', [3 3], 1.0);
    filtered_low = imfilter(filtered_median, h_low, 'replicate');

    % 3. High-pass filter (Laplacian sharpening)
    h_high = fspecial('laplacian', 0.2);
    lap = imfilter(filtered_low, h_high, 'replicate');
    filtered_high = filtered_low - lap;
    filtered_high(filtered_high < 0) = 0;
    filtered_high(filtered_high > 1) = 1;

    % The output is in [0, 1]. `skull_stripping` uses a threshold
    % of 30, which implies it expects an image in [0, 255]. We scale it up.
    % We convert to uint8 so skull_stripping works correctly with its threshold.
    output_img = im2uint8(filtered_high);
end
