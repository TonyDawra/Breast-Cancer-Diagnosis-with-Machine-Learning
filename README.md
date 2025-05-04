# Breast Cancer Diagnosis Machine Learning Project

This repository contains the implementation and report for a machine learning pipeline to classify breast tumors as benign or malignant using the Breast Cancer Wisconsin Diagnostic dataset. The project includes data preprocessing, feature selection, model training, evaluation, hyperparameter tuning, and result analysis.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Features & Targets](#features--targets)
3. [Project Structure](#project-structure)
4. [Environment & Dependencies](#environment--dependencies)
5. [Usage](#usage)

   * [Notebook Execution](#notebook-execution)
   * [Predictions CSV](#predictions-csv)
6. [Methodology](#methodology)

   * [Data Visualization](#data-visualization)
   * [Preprocessing](#preprocessing)
   * [Feature Selection](#feature-selection)
   * [Model Training & Evaluation](#model-training--evaluation)
   * [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Results](#results)
8. [Conclusion & Future Work](#conclusion--future-work)
9. [Authors](#authors)
10. [License](#license)

---

## Dataset

* **Source**: Breast Cancer Wisconsin Diagnostic dataset from the University of Wisconsin Hospitals, Madison.
* **Format**: 569 instances, 30 numeric features extracted from fine‑needle aspirate images, with diagnoses labeled as 'M' (malignant) or 'B' (benign).
* **Location**: Built‑in within scikit‑learn or available at \[UCI ML Repository].

---

## Features & Targets

* **Features (X)**: Radius, texture, perimeter, area, smoothness, compactness, concavity, etc. (30 attributes).
* **Target (y)**: Diagnosis encoded as 0 = benign, 1 = malignant.

---

## Project Structure

```
├── Final_project_AI_TonyDawra.ipynb  # Jupyter notebook pipeline
├── predictions.csv    # CSV of model predictions on test set
└── README.md          # Project documentation (this file)
```

---

## Environment & Dependencies

* Python 3.8+
* Jupyter Notebook
* Libraries:

  * numpy, pandas
  * scikit-learn
  * matplotlib, seaborn
  * imbalanced-learn
  * xgboost

Install via:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn imbalanced-learn xgboost
```

---

## Usage

### Notebook Execution

1. Launch Jupyter:

   ```bash
   jupyter notebook Final_project_AI_TonyDawra.ipynb
   ```
2. Run cells sequentially through:

   * Data loading and splitting
   * Visualization and preprocessing
   * PCA and feature engineering
   * Model training, cross-validation, and evaluation
   * Hyperparameter tuning
   * Metrics reporting and plots

### Predictions CSV

After running the notebook, the final test set predictions are saved to `predictions.csv`, containing:

```
| index | true_label | predicted_label |
```

---

## Methodology

### Data Visualization

* Examined class distribution (62.7% benign vs. 37.3% malignant).
* Correlation heatmap to detect feature redundancy.

### Preprocessing

1. **Encoding**: Label‑encoded targets (B=0, M=1).
2. **Resampling**: Random oversampling to balance classes.
3. **Scaling**: StandardScaler to zero mean and unit variance.

### Feature Selection

* **Principal Component Analysis (PCA)** reduced 30 features to 11 components capturing \~96% variance.

### Model Training & Evaluation

* Algorithms tested: Random Forest, XGBoost, SVM, KNN, Decision Tree.
* 5‑fold cross‑validation with metrics: accuracy, precision, recall, F1 score.
* Best base models: Regularized SVM (pre‑balance), Regularized XGBoost (post‑balance).

### Hyperparameter Tuning

* **XGBoost**: GridSearchCV optimized `learning_rate`, `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`, achieving \~98.25% accuracy.
* **SVM**: GridSearchCV on `C`, `kernel`, `gamma`, achieving \~94.8% accuracy.

---

## Results

* **Without Balancing**: SVM performed best; overfitting risk noted.
* **With Balancing**: Regularized XGBoost achieved top performance.
* **Threshold Analysis**: Explored decision thresholds at 0.2 (high recall) vs. 0.7 (high precision).

Key metrics for tuned XGBoost:

```
Accuracy: 98.25%
Precision: 97.9%
Recall: 98.6%
F1 Score: 98.2%
```

---

## Conclusion & Future Work

* XGBoost is the most effective classifier on this dataset.
* Future improvements:

  * Expand dataset scope (multi‑center data).
  * Explore deep learning models and interpretability (SHAP, LIME).
  * Deploy as a clinical decision support API (e.g., using Flask/FastAPI).

---

## Authors

* Tony Dawra
  
---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
