# train.py

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_fraud_10k.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model_pipeline.pkl")

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["is_fraud", "transaction_id"])
y = df["is_fraud"]

# -----------------------------
# Feature groups (IMPORTANT)
# -----------------------------
numeric_features = [
    "amount",
    "transaction_hour",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age"
]

categorical_features = ["merchant_category"]

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = Pipeline([
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Handle imbalance
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -----------------------------
# Model pipeline
# -----------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
print(classification_report(y_test, model.predict(X_test)))

# -----------------------------
# Save model
# -----------------------------
os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model saved successfully at:", MODEL_PATH)
