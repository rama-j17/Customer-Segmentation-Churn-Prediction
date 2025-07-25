import joblib, matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score)
from data_prep import load, make_pipeline
import numpy as np

X_train, X_test, y_train, y_test = load()
pipe = Pipeline([
    ('prep', make_pipeline()),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

model = pipe.fit(X_train, y_train)
joblib.dump(model, 'models/churn_logreg.joblib')

# Get feature names after transformation
feature_names = (
    model.named_steps['prep']
    .transformers_[0][1]  # OneHotEncoder
    .get_feature_names_out(['country', 'gender'])
    .tolist()
    + ['credit_score', 'age', 'tenure', 'balance',
       'products_number', 'credit_card', 'active_member', 'estimated_salary']
)

coefs = model.named_steps['clf'].coef_[0]
imp = pd.Series(coefs, index=feature_names).sort_values()

plt.figure(figsize=(8,6))
imp.plot(kind='barh', title='Feature Importance (Logistic Coefficients)')
plt.tight_layout()
plt.savefig('reports/visuals/feature_importance.png', dpi=120)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")

# Plots
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], '--');
plt.title('ROC Curve'); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.savefig('reports/visuals/roc_curve.png', dpi=120)

cm = confusion_matrix(y_test, y_pred)
plt.figure(); plt.imshow(cm, cmap='Blues');
plt.title('Confusion Matrix')
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha='center', va='center')
plt.xticks([0, 1], ['Stay', 'Churn'])
plt.yticks([0, 1], ['Stay', 'Churn'])
plt.savefig('reports/visuals/confusion_matrix.png', dpi=120)
