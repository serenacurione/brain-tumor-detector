function [seg_smooth_otsu, seg_multi, seg_edge] = segmentation(img)
%   Input:
%       img - Immagine in scala di grigi double [0, 1]
%   Outputs:
%       seg_smooth_otsu - Maschera binaria: Otsu preceduto da smoothing
%       seg_multi       - Matrice label: Segmentazione multi-livello (es. 3 classi)
%       seg_edge        - Maschera binaria: Otsu calcolato solo sui contorni (Edge-masked)

    %% 1. Sogliatura Globale con Pre-Smoothing (Otsu)
    img_smooth = imgaussfilt(img, 3);
    level_smooth = graythresh(img_smooth);
    seg_smooth_otsu = imbinarize(img_smooth, level_smooth);

    %% 2. Sogliatura Multi-livello (Multithresh)
    levels = multithresh(img, 2); 
    seg_multi = imquantize(img, levels);

    %% 3. Sogliatura Edge-Masked 
    % Utile per oggetti piccoli in grandi background (simmetria dell'istogramma)
    
    [Gmag, ~] = imgradient(img_smooth);
    max_grad = max(Gmag(:));
    edge_mask = Gmag > (0.55 * max_grad);
    
    img_masked = img .* double(edge_mask);
    
    valid_pixels_idx = img_masked > 0;
    level_edge = graythresh(img_masked(valid_pixels_idx));
    
    % Step E: Applicare la soglia trovata all'immagine globale
    seg_edge = imbinarize(img, level_edge);

end
