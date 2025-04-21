# Boxing Action Classifier

This project contains the script `boxing_ml_classifier.py`, which is used to classify boxing actions using various machine learning models, including:

- Random Forest
- Gradient Boosting
- XGBoost

## Description

The script applies supervised learning techniques to classify boxing-related actions based on extracted features. It evaluates and compares the performance of multiple ensemble classifiers to identify the most effective model.

## Input

- `boxing_balanced.pkl`: A pickle file containing a rebalanced dataset with:
  - `annotations`: A list of samples with keypoint data (`item['keypoint']`) and class labels (`item['label']`)
  - `split`: Dictionary with `xsub_train` and `xsub_test` keys listing the video names used for training and testing

## Output

- Trained classifier models saved as `.joblib` files:
  - `boxing_classifier_RandomForest.joblib`
  - `boxing_classifier_GradientBoosting.joblib`
  - `boxing_classifier_XGBoost.joblib` (if XGBoost is installed)

- Confusion matrix visualizations saved as PNG files:
  - `confusion_matrix_RandomForest.png`
  - `confusion_matrix_GradientBoosting.png`
  - `confusion_matrix_XGBoost.png` (if applicable)

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git](https://github.com/Omarion98/ML-Classifier-for-Boxing-Action-Detection.git
   python3 -m venv env
   pip install -U pip && pip install -r requirements.txt
   python boxing_ml_classifier.py
