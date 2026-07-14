# Figures Directory

This folder contains visualizations and plots generated during the ML final project.

## Purpose

The figures stored here are used for:

* Exploratory Data Analysis (EDA)
* Model evaluation
* Performance comparison
* Result visualization
* Interpretation of machine learning experiments

---

## Figure Types

### 1. Exploratory Data Analysis (EDA)

Includes visualizations that help understand dataset characteristics:

* Feature distributions
* Class distributions
* Correlation matrices
* Histograms
* Box plots

---

### 2. Model Performance

Includes plots generated after training machine learning models:

* Accuracy comparison
* Precision, recall, and F1-score comparisons
* Confusion matrices
* ROC curves
* Learning curves

---

### 3. Feature Analysis

Includes visualizations related to feature importance and selection:

* Feature importance plots
* PCA visualizations
* Feature contribution analysis

---

## File Naming Convention

Figures should follow a clear naming format:

```
<dataset>_<purpose>_<model>.png
```

Examples:

```
breast_cancer_confusion_matrix.png
adult_feature_importance_random_forest.png
covtype_model_comparison.png
mnist_pca_visualization.png
```

---

## Generation

Figures are generated automatically by the project scripts.

Example:

```bash
python run_all.py
```

Generated images are saved in this directory.

---
