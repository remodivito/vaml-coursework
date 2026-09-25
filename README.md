# Introduction

This report outlines the development and evaluation of a machine learning-based face detection system as part of our CSC3067 coursework. The coursework focuses on the implementation and testing of classification models and their integration into a comprehensive detection pipeline. The project emphasizes the use of feature extraction methods, classifier models, and parameter optimization to achieve robust and reliable performance.

The task involves training and testing three classification models—KNN, Random Forest, and SVM—using a dataset comprising labelled face and non-face images. A series of experiments are conducted to evaluate the impact of various parameters, preprocessing techniques, and feature extraction methods on classification accuracy and F1 score. These findings inform the selection of the optimal classifier for the face detection system.

To implement the detection system, the sliding window technique and Non-Maximum Suppression (NMS) are employed to identify faces across images at multiple scales while minimizing false positives. The final system integrates the selected classifier and employs parameter-tuned settings to maximize detection accuracy.

This report details each stage of the project, including classifier comparison, preprocessing, feature extraction, data augmentation, and the sliding window implementation. Emphasis is placed on justifying decisions with quantitative results and aligning system design with the module’s objectives.
