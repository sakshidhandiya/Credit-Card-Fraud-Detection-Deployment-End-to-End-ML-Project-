# app.py

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Load model
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model_pipeline.pkl")

model = joblib.load(MODEL_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0"
)

# -----------------------------
# Input schema
# -----------------------------
class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.dict()])
    prob = model.predict_proba(data)[0][1]

    threshold = 0.3
    return {
        "fraud_probability": round(float(prob), 4),
        "is_fraud": int(prob >= threshold)
    }
