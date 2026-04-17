# Brain Tumor MRI Detector

A MATLAB-based automated system for the classification of brain tumors from MRI images. The system implements a complete pipeline from image preprocessing to multiclass classification using Support Vector Machines (SVM).

## Overview

The project aims to distinguish between different types of brain tumors (Glioma, Meningioma, Pituitary) and healthy brains (No tumor) using standard image processing techniques and machine learning.

## Pipeline Architecture

The system follows a sequential pipeline:

1.  **Preprocessing**: Image resizing, noise reduction (Median & Gaussian filters), and edge sharpening (Laplacian filter).
2.  **Segmentation**:
    - **Skull Stripping**: Removing non-brain tissues using morphological operations.
    - **Tumor Extraction**: Isolating the tumor mass using Otsu's thresholding and connected component analysis.
3.  **Feature Extraction**: Calculation of 7 statistical features from the segmented area:
    - Mean Intensity
    - Standard Deviation
    - Entropy
    - 3rd Moment (Skewness)
    - 4th Moment (Kurtosis)
    - Smoothness
    - Uniformity
4.  **Classification**: Multiclass SVM (ECOC approach) with automated **Grid Search** to find the best kernel (Linear, RBF, Polynomial, Quadratic) and hyperparameters (C, Gamma).

## Dataset Structure

The system expects the dataset in the following format:

```text
dataset/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── ...
```

## Requirements

- **MATLAB** (R2021a+)
- **Image Processing Toolbox**
- **Statistics and Machine Learning Toolbox**

## How to Use

1.  Place your dataset in the `dataset/` folder.
2.  Open MATLAB and navigate to the project directory.
3.  Run the main script:
    ```matlab
    main
    ```
4.  To visualize intermediate results, run:
    ```matlab
    save_segmentation_results
    ```

## Performance

The system evaluates performance using:

- Accuracy
- Sensitivity (Recall)
- Specificity
- Precision
- F1-Score

For detailed technical information, see [DOCUMENTATION.md](file:///Users/nicolobruno/Coding-Projects/brain-tumor-detector/DOCUMENTATION.md).
