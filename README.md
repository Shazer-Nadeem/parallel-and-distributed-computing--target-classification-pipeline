# PDC Classification Project

## Overview

This repository contains a complete machine learning pipeline to solve a classification problem using the PDC dataset. The pipeline includes data preprocessing, model training using multiple classifiers, evaluation based on relevant metrics, and persistence of the best-performing model and preprocessing pipeline.

The aim is to predict the `target` variable based on numerical input features extracted from the dataset.

---

## Dataset Description

* **Filename**: `pdc_dataset_with_target.csv`
* **Problem Type**: Classification
* **Target Variable**: `target`
* **Features**: Numerical features
* **Missing Values**: Present and handled in preprocessing

---

## Environment Setup

Make sure you have Python 3.7 or later installed. You can install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

---

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/pdc-classification.git
cd pdc-classification
```

2. Place the dataset `pdc_dataset_with_target.csv` in the project root.

3. Run your processing and training script (e.g., a notebook or script file that you implement using the logic from this repo).

Example if using a Python script:

```bash
python pdc_pipeline.py
```

---

## Project Workflow

The project consists of the following major steps:

1. **Data Loading**

   * Load the dataset and inspect basic properties.

2. **Data Splitting**

   * Split into training and testing sets (stratified).

3. **Preprocessing**

   * Missing value imputation (mean)
   * Feature scaling (StandardScaler)

4. **Model Training**

   * Train and evaluate:

     * Logistic Regression
     * Random Forest Classifier
     * XGBoost Classifier

5. **Evaluation**

   * Metrics used:

     * Accuracy
     * F1 Score
     * ROC AUC Score
     * Confusion Matrix
     * ROC Curve

6. **Model Selection and Persistence**

   * The best model is selected based on weighted F1 score.
   * The model and preprocessing pipeline are saved as:

     * `best_model.pkl`
     * `preprocessing_pipeline.pkl`

---

## Evaluation Metrics

Each classifier is evaluated using:

* Accuracy
* F1 Score (Weighted)
* ROC AUC Score
* Confusion Matrix
* Classification Report
* ROC Curve Plot

---

## Directory Structure

```
pdc-classification/
├── pdc_dataset_with_target.csv
├── pdc_pipeline.py                # (Your main script or notebook)
├── best_model.pkl
├── preprocessing_pipeline.pkl
└── README.md
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or suggestions, feel free to contact \[[your-email@example.com](mailto:your-email@example.com)] or open an issue.

---

## Acknowledgments

* [Scikit-learn](https://scikit-learn.org/)
* [XGBoost](https://xgboost.readthedocs.io/)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)
