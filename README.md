# customer-churn-prediction
Built a predictive model to identify customers likely to churn using customer behavior and transaction data. Applied data visualization and machine learning techniques to enhance business decision-making.
Project Title :- 
Customer Churn Prediction using Machine Learning

Problem Statement:-
Telecom companies face significant revenue loss due to customer churn. The goal of this project is to build a machine learning model that predicts whether a customer will leave the service (churn) based on usage patterns, contract details, and demographics.

Dataset:- 
Source: [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Size: 7,043 customer records, 21 features
Target Variable: Churn (Yes/No)

Tech Stack:- 
Languages: Python
Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
Tools: Jupyter Notebook 

Steps Followed:-
Data Cleaning:
Removed unnecessary columns
Converted data types
Handled missing values in TotalCharges

Exploratory Data Analysis (EDA):
Visualized churn patterns by contract type, tenure, monthly charges

Feature Engineering:
Label encoding for binary features
One-hot encoding for multi-class features
Feature scaling with StandardScaler

Model Training:
Compared Logistic Regression, Random Forest, XGBoost

Evaluation:
Accuracy, Precision, Recall, F1 Score, Confusion Matrix

Feature Importance:
Identified top churn predictors

 Results
 Best Model: XGBoost
Accuracy: ~85%
Key Insights:
Month-to-month contracts have higher churn
Lower tenure customers are more likely to churn
Higher monthly charges increase churn probability

Project Structure:- 
customer-churn-prediction/
│── data/                # Dataset
│── notebooks/           # Jupyter notebooks for EDA & modeling
│── scripts/             # Python scripts
│── outputs/             # Model files, plots
│── README.md
│── requirements.txt
│── churn_model.pkl 

Code :- 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.head())


df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)


for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() == 2:
        df[col] = LabelEncoder().fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col])


X = df.drop('Churn', axis=1)
y = df['Churn']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)




log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)


rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)


def evaluate_model(name, y_true, y_pred):
    print(f"Model: {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("XGBoost", y_test, y_pred_xgb)


importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.show()


import joblib
joblib.dump(rf_model, "customer_churn_model.pkl")
print("Model saved as customer_churn_model.pkl")
