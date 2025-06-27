Stroke Risk Detection Using Retinal Fundus Images
This project focuses on detecting potential signs of stroke by analyzing retinal fundus images using deep learning. It combines vessel segmentation, vascular feature extraction, and narrowing classification to assess stroke risk in a non-invasive and efficient way.

Project Pipeline
The flow of the project includes the following steps:

Image Preprocessing

Enhance retinal images and normalize input for segmentation.

Vessel Segmentation

A custom-trained Improved U-Net model is used to segment retinal blood vessels from fundus images.

Feature Extraction

The segmented vessel masks are processed using skeletonization to extract vessel width and pattern.

Narrowing Detection

A rule-based narrowing classifier analyzes vessel width to detect possible arterial narrowing, which may indicate stroke risk.

Risk Assessment

Final output highlights narrowed regions and provides a risk label.

