"""
Disease Prediction from Medical Data (Breast Cancer Classification)
Task 4 - CodeAlpha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
print("Loading Breast Cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')  # 0 = malignant, 1 = benign

print(f"Total samples: {X.shape[0]}")
print(f"Total features: {X.shape[1]}")
print(f"Class distribution:\n{y.value_counts()}")

# -----------------------------
# 2. Explore Data
# -----------------------------
print("\nFirst 5 rows:")
print(X.head())

plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title('Class Distribution (0 = Malignant, 1 = Benign)')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()
print("Saved class_distribution.png")

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 4. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

results = {}

print("\nTraining models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

# -----------------------------
# 6. Compare Models (Bar Chart)
# -----------------------------
metrics_df = pd.DataFrame({
    name: {
        "Accuracy": r["accuracy"],
        "Precision": r["precision"],
        "Recall": r["recall"],
        "F1-Score": r["f1"],
        "ROC-AUC": r["roc_auc"]
    } for name, r in results.items()
}).T

plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar', ax=plt.gca())
plt.title('Model Comparison Across Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("\nSaved model_comparison.png")

# -----------------------------
# 7. Pick Best Model (by F1-Score)
# -----------------------------
best_name = metrics_df["F1-Score"].idxmax()
best_result = results[best_name]
print(f"\nBest model based on F1-Score: {best_name}")

# -----------------------------
# 8. Confusion Matrix (Best Model)
# -----------------------------
cm = confusion_matrix(y_test, best_result["y_pred"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("Saved confusion_matrix.png")

# -----------------------------
# 9. ROC Curve (All Models)
# -----------------------------
plt.figure(figsize=(7, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {r['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()
print("Saved roc_curve.png")

# -----------------------------
# 10. Feature Importance (Random Forest)
# -----------------------------
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title('Top 10 Important Features (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()
print("Saved feature_importance.png")

# -----------------------------
# 11. Classification Report (Best Model)
# -----------------------------
print(f"\nClassification Report - {best_name}:")
print(classification_report(y_test, best_result["y_pred"],
                             target_names=['Malignant', 'Benign']))

# -----------------------------
# 12. Save the Best Model
# -----------------------------
import joblib
joblib.dump(best_result["model"], 'disease_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(f"\nBest model ({best_name}) saved as disease_prediction_model.pkl")
print("Scaler saved as scaler.pkl")

print("\nDone! Check the folder for generated images and model file.")
