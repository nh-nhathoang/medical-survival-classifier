# Predicting survival of Patients with Heart Failure

## Project Overview
Heart failure is a serious condition that affects millions worldwide. It occurs when the heart can’t pump blood effectively, leading to symptoms like fatigue, breathlessness, and fluid buildup. While the prognosis can vary greatly, certain clinical factors can help predict a patient's risk of survival. This project uses machine learning to analyze these factors and improve the accuracy of these predictions.

This project applies machine learning to predict survival in patients with heart failure, based on their clinical data. By predicting patient outcomes, this project aims to support better treatment decisions and improve survival rates.

## Problem Setup
Our goal is to predict whether a patient with heart failure will survive. This is a **binary classification task**:
- **Labels**: 0 = survived, 1 = deceased
- **Features**: Clinical variables like age, blood pressure, and ejection fraction (percentage of blood leaving the heart per contraction).
  
## Dataset

The dataset, from the Heart Failure Clinical Records Dataset. UCI Machine Learning Repository. https://doi.org/10.24432/C5Z89R, includes 299 patient records with 12 clinical features.

## Model Selection
We evaluate two primary models:
1. **Logistic Regression**: A simple yet effective model for binary classification, well-suited for healthcare applications due to its interpretability.

2. **Support Vector Classifier (SVC)**: popular for handling binary classification tasks with complex decision boundaries. SVC works by maximizing the margin between classes.

We split the data into 80% for training/validation and 20% for testing. For model validation, we use 4-fold cross-validation to avoid overfitting and get a more accurate performance assessment.

## Setup

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

```bash
python main.py
```

## Results

Both models have similar AUC but LR generalizes better on the minority class (deceased), likely because SVC overfits on the small dataset. Accuracy is not reported — misleading on imbalanced data.