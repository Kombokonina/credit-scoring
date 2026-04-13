# Credit Scoring Model

A model for identifying borrowers with high default risk based on credit history, income, and payment behavior.

## Results

| Model | ROC AUC |
|-------|---------|
| Logistic Regression | 0.800 |
| Logistic Regression (balanced) | 0.850 |
| GradientBoosting (5 features) | 0.808 |
| GradientBoosting (all features) | 0.850 |
| CatBoost | 0.848 |
| CatBoost (Optuna tuned) | 0.855 |
| Pipeline + GradientBoosting | 0.861 |
| **Pipeline + CatBoost (Optuna)** | **0.864** |

## Key Findings

- Class imbalance (6.7% defaults) makes accuracy misleading — baseline model showed 93% accuracy but only 15% recall on defaulters
- Applying `class_weight='balanced'` increased recall from 15% to 59%
- sklearn Pipeline with SimpleImputer preserved ~30K rows that dropna() removed, improving ROC AUC by 1%
- K-Means clustering without the target variable identified a group with 48% default rate
- Top 3 features by importance: RevolvingUtilizationOfUnsecuredLines (25%), DebtRatio (12%), age (12%)

## Stack

Python, Pandas, Scikit-learn, CatBoost, Optuna, Matplotlib

## Data

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150K borrower records with 10 features.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook notebooks/main.ipynb
```