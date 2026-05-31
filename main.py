import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, f1_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# data loaded via UCI API (see fetch_ucirepo below)
# if you want to load from local CSV instead, uncomment below and comment out the fetch_ucirepo line
# data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
heart_failure = fetch_ucirepo(id=519)

# correlation heatmap
data = heart_failure.data.features.copy().join(heart_failure.data.targets)
print(data.info())

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap='coolwarm',
            square=True, cbar_kws={'shrink': 0.75})
plt.title('Correlation Matrix of Heart Failure Clinical Records')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()

# keep only 5 features most correlated with the target
X = heart_failure.data.features[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]
y = heart_failure.data.targets.squeeze() 

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train_val)  # scale only on training data
X_test = scaler.transform(X_test)                # apply the same scaler to test

# hyperparameter search: use AUC, not accuracy, because the data is imbalanced
lr_params = [
    {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']},
    {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1'], 'solver': ['liblinear']},
]
lr_model = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), lr_params,
                        cv=5, scoring='roc_auc').fit(X_train_val, y_train_val).best_estimator_

svc_params = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
}
svc_model = GridSearchCV(SVC(probability=True, class_weight='balanced'), svc_params, cv=5, scoring='roc_auc').fit(X_train_val, y_train_val).best_estimator_


# 4-fold cross-validation
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

results = {
    'lr':  {'train_loss': [], 'val_loss': [], 'auc': [], 'f1': [], 'recall': []},
    'svc': {'train_loss': [], 'val_loss': [], 'auc': [], 'f1': [], 'recall': []},
}

for train_idx, val_idx in kf.split(X_train_val, y_train_val):
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    for name, model in [('lr', lr_model), ('svc', svc_model)]:
        model.fit(X_train, y_train)
        proba_train = model.predict_proba(X_train)
        proba_val   = model.predict_proba(X_val)
        y_pred      = model.predict(X_val)

        results[name]['train_loss'].append(log_loss(y_train, proba_train))
        results[name]['val_loss'].append(log_loss(y_val, proba_val))
        results[name]['auc'].append(roc_auc_score(y_val, proba_val[:, 1]))
        results[name]['f1'].append(f1_score(y_val, y_pred))
        results[name]['recall'].append(recall_score(y_val, y_pred))


# final evaluation on held-out test set
for model in [lr_model, svc_model]:
    model.fit(X_train_val, y_train_val)

def test_metrics(model):
    proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    return {
        'proba':       proba[:, 1],   
        'y_pred':      y_pred,        
        'test_loss':   log_loss(y_test, proba),
        'test_auc':    roc_auc_score(y_test, proba[:, 1]),
        'test_f1':     f1_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
    }

lr_test  = test_metrics(lr_model)
svc_test = test_metrics(svc_model)

#confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, res) in zip(axes, [('Logistic Regression', lr_test), ('SVC', svc_test)]):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Survived', 'Deceased'],
                yticklabels=['Survived', 'Deceased'])
    ax.set_title(f'{name}\nConfusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.close()
print("Saved: confusion_matrices.png")

# ROC curves
plt.figure(figsize=(7, 6))
for name, res, color in [('Logistic Regression', lr_test, 'steelblue'), ('SVC', svc_test, 'tomato')]:
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {res['test_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300)
plt.close()
print("Saved: roc_curve.png")

# print results
def avg(lst): return np.mean(lst)

print(f"{'Metric':<30}{'Logistic Regression':<22}{'SVC'}")
print(f"{'Train Log Loss':<30}{avg(results['lr']['train_loss']):<22.4f}{avg(results['svc']['train_loss']):.4f}")
print(f"{'Val Log Loss':<30}{avg(results['lr']['val_loss']):<22.4f}{avg(results['svc']['val_loss']):.4f}")
print(f"{'Val AUC-ROC':<30}{avg(results['lr']['auc']):<22.4f}{avg(results['svc']['auc']):.4f}")
print(f"{'Val F1 (deceased)':<30}{avg(results['lr']['f1']):<22.4f}{avg(results['svc']['f1']):.4f}")
print(f"{'Val Recall (deceased)':<30}{avg(results['lr']['recall']):<22.4f}{avg(results['svc']['recall']):.4f}")
print(f"{'Test Log Loss':<30}{lr_test['test_loss']:<22.4f}{svc_test['test_loss']:.4f}")
print(f"{'Test AUC-ROC':<30}{lr_test['test_auc']:<22.4f}{svc_test['test_auc']:.4f}")
print(f"{'Test F1 (deceased)':<30}{lr_test['test_f1']:<22.4f}{svc_test['test_f1']:.4f}")
print(f"{'Test Recall (deceased)':<30}{lr_test['test_recall']:<22.4f}{svc_test['test_recall']:.4f}")