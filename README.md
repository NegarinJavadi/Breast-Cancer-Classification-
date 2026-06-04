This project focuses on the classification of breast tumors as either benign (non-cancerous) or malignant(cancerous) using machine learning techniques. The work is inspired by the research paper "On Breast Cancer Detection: An Application of Machine Learning Algorithms on the Wisconsin Diagnostic Dataset" and uses the well-known Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository.

Number of samples:569 patients
Number of features:30 numerical features

Classes:
  Benign (B)
  Malignant (M)

The dataset contains measurements computed from digitized images of fine needle aspirate (FNA) tests of breast masses. Features describe characteristics of cell nuclei such as:
  Radius
  Texture
  Perimeter
  Area
  Smoothness
  Compactness
  Concavity
  Symmetry
  Fractal Dimension

Several machine learning algorithms were trained and compared, including:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Random Forest
5. Gradient Boosting

To ensure robust model evaluation, the following techniques and metrics were used:
  Train/Test Split
  5-Fold Cross Validation
  Accuracy
  Precision
  Recall
  F1 Score
  ROC Curve
  AUC-ROC

AUC-ROC was selected as a key metric because it measures how effectively a model can separate malignant and benign cases across different decision thresholds.

The models achieved strong classification performance on the Wisconsin Diagnostic Dataset.
  Test Accuracy: ~95–99%
  AUC-ROC: > 0.98

These results demonstrate that machine learning algorithms can successfully identify patterns associated with breast cancer diagnosis and provide highly accurate predictions on unseen patient data.

https://arxiv.org/abs/1711.07831

Conclusion:
This project demonstrates the effectiveness of machine learning techniques for breast cancer diagnosis using clinical imaging-derived features. By comparing multiple classification algorithms under the same evaluation framework, the project highlights how data-driven approaches can support medical decision-making and contribute to early disease detection.
