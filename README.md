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


joblib.dump(rf_model, "customer_churn_model.pkl")
print("Model saved as customer_churn_model.pkl")
