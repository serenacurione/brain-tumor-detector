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

    
    % High-pass filter (Laplacian sharpening)
    h_high = fspecial('laplacian', 0.2);
    lap = imfilter(img, h_high, 'replicate');
    filtered_high = img - lap;
    filtered_high = mat2gray(filtered_high);
    
    % Median filter (salt-and-pepper / high-freq noise removal)
    filtered_median = medfilt2(filtered_high, [3 3]);
    filtered_median = mat2gray(filtered_median);

    % Low-pass filter (Gaussian smoothing)
    h_low = fspecial('gaussian', [5 5], 1.5);
    filtered_low = imfilter(filtered_median, h_low, 'replicate');
    filtered_low = mat2gray(filtered_low);


    % The output is in [0, 1]. `skull_stripping` uses a threshold
    % of 30, which implies it expects an image in [0, 255]. We scale it up.
    % We convert to uint8 so skull_stripping works correctly with its threshold.
    output_img = im2uint8(filtered_low);
end
