# Predicting the survival of Patients with Heart Failure

## Project Overview
Heart failure is a serious condition that affects millions worldwide. It occurs when the heart can’t pump blood effectively, leading to symptoms like fatigue, breathlessness, and fluid buildup. While the prognosis can vary greatly, certain clinical factors can help predict a patient's risk of survival. This project uses machine learning to analyze these factors and improve the accuracy of these predictions.

This project applies machine learning to predict survival in patients with heart failure, based on their clinical data. By predicting patient outcomes, this project aims to support better treatment decisions and improve survival rates.

## Problem Setup
Our goal is to predict whether a patient with heart failure will survive. This is a **binary classification task**:
- **Labels**: 0 = survived, 1 = deceased
- **Features**: Clinical variables, such as age, blood pressure, and ejection fraction (percentage of blood leaving the heart per contraction).
  
## Dataset

The dataset, from the Heart Failure Clinical Records Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5Z89R, includes 299 patient records with 12 clinical features.

## Model Selection
We evaluate two primary models:
1. **Logistic Regression**: A simple yet effective model for binary classification, well-suited for healthcare applications due to its interpretability.

2. **Support Vector Classifier (SVC)**: popular for handling binary classification tasks with complex decision boundaries. SVC works by maximizing the margin between classes.

We split the data into 80% for training/validation and 20% for testing. For model validation, we use 4-fold cross-validation to avoid overfitting and get a more accurate performance assessment.

## Evaluation
- **Recall**: most critical in the medical because it minimizes false negatives
- **F1 score**: balances recall against precision
- **AUC-ROC**: threshold-free measure of ranking ability

## Setup

```bash
pip install pandas numpy scikit-learn seaborn matplotlib ucimlrepo
```

```bash
python main.py
```

## Results

Both models achieve similar AUC-ROC (0.89), but Logistic Regression generalizes better to the minority class (deceased), with higher test recall (0.684 vs 0.579) and F1 (0.684 vs 0.629). SVC might overfit the small dataset, because it produces good probabilities but a suboptimal decision boundary. 

LR is the preferred model for this task for two reasons. First, it outperforms SVC on the metrics that matter most clinically, including recall and F1 on the deceased class. Second, unlike SVC with a non-linear kernel, LR always provides a clean explanation of what drives each prediction via its coefficients. In medical, it is important to understand why a patient is flagged as high risk.

The LR coefficient plot confirms the model learned meaningful patterns: `serum_creatinine` and `ejection_fraction` are the strong predictors. 

`time`, which is follow-up time, is highly correlated but it is a consequence of the target, not the cause. Therefore, it need to be removed.