# Credit-Card-Fraud-Detection-Deployment-End-to-End-ML-Project-
End-to-end credit card fraud detection system built using ensemble machine learning models and deployed as a real-time FastAPI service. The solution handles class imbalance, optimizes fraud recall, and supports probability-based decisioning.
📌 Project Overview

Credit card fraud is a high-impact, low-frequency problem where missing fraudulent transactions can result in significant financial losses.
This project presents an end-to-end machine learning solution for detecting fraudulent credit card transactions, covering the entire lifecycle — from data preprocessing and model building to real-time deployment using FastAPI.

The solution is designed with real-world constraints in mind:

Highly imbalanced data

Cost-sensitive decision making

Model interpretability vs performance trade-offs

Production-ready deployment practices

🎯 Objectives

Build robust machine learning models to identify fraudulent transactions

Handle severe class imbalance effectively

Compare baseline, tree-based, and ensemble models

Optimize decision thresholds for fraud detection

Deploy the final model as a real-time REST API

📊 Dataset Description

Type: Synthetic credit card transaction data (privacy-safe)

Records: ~10,000

Target Variable: is_fraud

0 → Legitimate transaction

1 → Fraudulent transaction

Class Distribution: Highly imbalanced (fraud ≈ 4–5%)

Key Features
Feature	Description
amount	Transaction amount
transaction_hour	Hour of transaction (0–23)
merchant_category	Type of merchant
foreign_transaction	International transaction flag
location_mismatch	Billing vs transaction location mismatch
device_trust_score	Device trust score (0–100)
velocity_last_24h	Number of transactions in last 24 hours
cardholder_age	Age of cardholder
🧠 Machine Learning Approach
1️⃣ Baseline Model – Logistic Regression

Served as an interpretable baseline

Provided insight into key fraud drivers

Highlighted the limitations of linear models on complex fraud patterns

2️⃣ Decision Tree

Captured non-linear rules and interactions

Achieved very high recall but suffered from overfitting and low precision

3️⃣ Random Forest (Bagging Ensemble)

Improved stability over a single decision tree

Achieved strong recall with better precision

Demonstrated the benefit of ensemble averaging

4️⃣ XGBoost (Final Model – Boosting Ensemble)

Best overall performance

Excellent handling of non-linear interactions

Optimized using class imbalance handling (scale_pos_weight)

Selected as the final production model

📊 Dataset Description

Type: Synthetic credit card transaction data (privacy-safe)

Records: ~10,000

Target Variable: is_fraud

0 → Legitimate transaction

1 → Fraudulent transaction

Class Distribution: Highly imbalanced (fraud ≈ 4–5%)

Key Features
Feature	Description
amount	Transaction amount
transaction_hour	Hour of transaction (0–23)
merchant_category	Type of merchant
foreign_transaction	International transaction flag
location_mismatch	Billing vs transaction location mismatch
device_trust_score	Device trust score (0–100)
velocity_last_24h	Number of transactions in last 24 hours
cardholder_age	Age of cardholder
🧠 Machine Learning Approach
1️⃣ Baseline Model – Logistic Regression

Served as an interpretable baseline

Provided insight into key fraud drivers

Highlighted the limitations of linear models on complex fraud patterns

2️⃣ Decision Tree

Captured non-linear rules and interactions

Achieved very high recall but suffered from overfitting and low precision

3️⃣ Random Forest (Bagging Ensemble)

Improved stability over a single decision tree

Achieved strong recall with better precision

Demonstrated the benefit of ensemble averaging

4️⃣ XGBoost (Final Model – Boosting Ensemble)

Best overall performance

Excellent handling of non-linear interactions

Optimized using class imbalance handling (scale_pos_weight)

Selected as the final production model

📈 Model Performance (Final XGBoost)

Recall (Fraud): ~98%

Precision (Fraud): ~98%

ROC-AUC: ~0.99

False Negatives: Extremely low (critical for fraud use cases)

⚠️ Accuracy is not used as the primary metric due to class imbalance.
Recall, precision, and ROC-AUC drive all decisions.

⚖️ Fraud Decision Strategy

Instead of relying purely on accuracy, the project uses probability-based decisioning:

Low probability → Legitimate transaction

Medium probability → Manual review / step-up authentication

High probability → Fraud (block transaction)

This mirrors real-world banking and fintech fraud systems.

🚀 Deployment Architecture
Tech Stack

Modeling: Python, scikit-learn, XGBoost

Preprocessing: sklearn Pipelines & ColumnTransformer

API: FastAPI

Serialization: joblib

Why FastAPI?

High performance

Automatic request validation

Interactive API documentation (Swagger UI)

Production-friendly architecture
