# Optimizing Crop Selection Using Machine Learning for Sustainable Agriculture in Egypt

🎓 Graduation Project  
📄 Published Research Paper

This repository contains the full implementation of a **machine learning–based crop recommendation system**
developed as a graduation project and published in an Egyptian scientific journal.

The project applies supervised machine learning techniques to recommend the most suitable crop based on
environmental and soil parameters under Egyptian agricultural conditions.

---

## Problem Statement

Crop selection in Egypt faces increasing challenges due to:
- Climate variability
- Water scarcity
- Soil degradation

Traditional farming practices are insufficient to handle the complexity of modern environmental data.
This project proposes a **data-driven decision-support system** to improve crop selection and sustainability.

---

## Dataset & Features

Each data sample includes:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Soil pH
- Temperature (°C)
- Humidity (%)
- Rainfall (mm)

Data sources include FAO, Kaggle, and Egyptian Agricultural Research Center datasets.

---

## Methodology

### Models Evaluated
- Decision Tree (DT)
- Support Vector Machine (SVM)
- Linear Regression (LR)
- **Random Forest (Proposed Model)**

### Preprocessing & Enhancements
- Missing value imputation
- MinMaxScaler normalization
- Label Encoding
- SMOTE for class imbalance
- Feature engineering
- GridSearchCV with 5-fold cross-validation

### Interpretability
- SHAP (SHapley Additive Explanations) used to analyze feature importance and model behavior.

---

## Results

The Random Forest model achieved:
- **Accuracy:** 100%
- **Precision:** 1.00
- **Recall:** 1.00
- **F1-score:** 1.00
- **Cross-validation accuracy:** 99.38%

The model demonstrated strong generalization, robustness against overfitting, and interpretability,
making it suitable for real-world agricultural advisory systems.

---

## Repository Structure

```text
src/            Core ML pipeline
notebooks/      Experimental & analysis notebooks
data/           Raw and processed datasets
results/        Evaluation outputs and figures
paper/          Published research paper (PDF)

```

---

## Running the Project

```text
pip install -r requirements.txt
python src/main_pipeline.py
```
