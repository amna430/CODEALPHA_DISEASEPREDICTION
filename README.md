# CodeAlpha_DiseasePrediction

## 📌 Task
CodeAlpha Machine Learning Internship — Task 4: Disease Prediction from Medical Data

## 🎯 Objective
Predict the possibility of disease (malignant vs benign tumor diagnosis) based on structured medical data, using classification algorithms.

## 🧠 Approach
- Used the **Breast Cancer Wisconsin dataset** (built into Scikit-learn)
- Performed feature scaling using `StandardScaler`
- Trained and compared three classification models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Evaluated each model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Selected the best-performing model based on F1-Score
- Visualized results with a confusion matrix, ROC curves, and feature importance ranking
- Saved the best model and the scaler for future use/inference

## 📂 Project Structure
```
CodeAlpha_DiseasePrediction/
│
├── disease_prediction.py           # Main script: data loading, training, evaluation
├── disease_prediction_model.pkl    # Saved best-performing model
├── scaler.pkl                      # Saved feature scaler
├── class_distribution.png          # Distribution of malignant vs benign cases
├── model_comparison.png            # Metric comparison across the three models
├── confusion_matrix.png            # Confusion matrix of the best model
├── roc_curve.png                   # ROC curve comparison across models
├── feature_importance.png          # Top 10 most important features (Random Forest)
└── README.md
```

## 📊 Results
- Compared Logistic Regression, Random Forest, and SVM on the same train/test split
- Best model selected automatically based on F1-Score
- Achieved strong performance across all metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Feature importance analysis highlighted the most predictive medical features (e.g., worst radius, worst perimeter, worst concave points)

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

## ▶️ How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
python disease_prediction.py
```

## 📈 Possible Extensions
- Apply to other medical datasets (Heart Disease, Diabetes)
- Add hyperparameter tuning (GridSearchCV) for further optimization
- Try additional algorithms like XGBoost
