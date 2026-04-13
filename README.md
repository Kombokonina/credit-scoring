# Модель кредитного скоринга / Credit Scoring Model

Модель для выявления заёмщиков с высоким риском дефолта на основе данных о кредитной истории, доходах и поведении.

## Результаты

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

## Основные выводы

- Дисбаланс классов (6.7% дефолтов) делает accuracy бесполезной метрикой — базовая модель показывала 93% accuracy при recall дефолтов всего 15%
- Применение `class_weight='balanced'` увеличило recall с 15% до 59%
- sklearn Pipeline с SimpleImputer сохранил ~30K строк, которые dropna() удалял, что улучшило ROC AUC на 1%
- K-Means кластеризация без целевой переменной выявила группу с 48% уровнем дефолтов
- Топ-3 признака по важности: RevolvingUtilizationOfUnsecuredLines (25%), DebtRatio (12%), age (12%)

## Стек

Python, Pandas, Scikit-learn, CatBoost, Optuna, Matplotlib

## Данные

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150K записей заёмщиков с 10 признаками.

## Запуск

```bash
pip install -r requirements.txt
jupyter notebook notebooks/credit_scoring.ipynb
```