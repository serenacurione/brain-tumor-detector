function [binary_mask, tumor] = tumor_extraction(img, brain, varargin)

  % Estraiamo i pixel del cervello ignorando lo sfondo nero (0) 
  % in modo da non sbilanciare il calcolo della soglia di Otsu
  brain_pixels = brain(brain > 0);

  if isempty(brain_pixels)
      binary = false(size(brain));
  else
      % Calcoliamo la soglia dinamica con il metodo di Otsu [0, 1]
      level = graythresh(brain_pixels);
      
      % graythresh restituisce un valore normalizzato, lo riportiamo a 0-255
      threshold = level * 255;
      
      % Binarizziamo l'immagine usando la nuova soglia adattiva
      binary = double(brain) > threshold;
  end

  % morphological operations
  se = strel('diamond', 3);
  binary_eroded = imerode(binary, se);
  CC = bwconncomp(binary_eroded);
  binary_comp = false(size(binary_eroded));
  if CC.NumObjects > 0
      numPixels = cellfun(@numel, CC.PixelIdxList);
      [~, idx] = max(numPixels);
      binary_comp(CC.PixelIdxList{idx}) = true;
  end
  binary_dilated = imdilate(binary_comp, se);
  binary_mask = imfill(binary_dilated, "holes");

  % tumor extraction
  tumor = double(brain) .* double(binary_mask);
  tumor = uint8(tumor);

end