from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import torch
from catboost import Pool
from fastapi import FastAPI
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import ALL_FEATURE_COLUMNS, CATEGORICAL_COLUMNS, MODELS_DIR, NUMERICAL_COLUMNS  # noqa: E402
from src.mlp_pipeline import TabularMLP  # noqa: E402


class CreditDefaultRequest(BaseModel):
    limit_bal: int = Field(..., ge=0)
    sex: str
    education: str
    marriage: str
    age: int = Field(..., ge=18)
    pay_0: int
    pay_2: int
    pay_3: int
    pay_4: int
    pay_5: int
    pay_6: int
    bill_amt1: float
    bill_amt2: float
    bill_amt3: float
    bill_amt4: float
    bill_amt5: float
    bill_amt6: float
    pay_amt1: float
    pay_amt2: float
    pay_amt3: float
    pay_amt4: float
    pay_amt5: float
    pay_amt6: float


def payload_to_frame(payload: CreditDefaultRequest) -> pd.DataFrame:
    return pd.DataFrame([payload.model_dump()])


@lru_cache
def load_mlp_assets():
    metadata = joblib.load(MODELS_DIR / "mlp_metadata.joblib")
    model = TabularMLP(
        cardinalities=metadata["cardinalities"],
        num_numerical_features=len(metadata["numerical_columns"]),
        hidden_dims=metadata["best_config"]["hidden_dims"],
        dropout=metadata["best_config"]["dropout"],
    )
    state_dict = torch.load(MODELS_DIR / "mlp_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return metadata, model


@lru_cache
def load_tree_model():
    return joblib.load(MODELS_DIR / "catboost_model.joblib")


def encode_for_mlp(frame: pd.DataFrame, metadata: dict):
    categorical_values = []
    for col in metadata["categorical_columns"]:
        mapping = metadata["category_maps"][col]
        categorical_values.append(mapping.get(str(frame.iloc[0][col]), 0))
    numerical_values = []
    for col in metadata["numerical_columns"]:
        mean = metadata["numerical_means"][col]
        std = metadata["numerical_stds"][col]
        numerical_values.append((float(frame.iloc[0][col]) - mean) / std)

    cat_tensor = torch.tensor([categorical_values], dtype=torch.long)
    num_tensor = torch.tensor([numerical_values], dtype=torch.float32)
    return cat_tensor, num_tensor


def prediction_response(probability: float) -> dict:
    label = "default" if probability >= 0.5 else "no_default"
    return {"prediction": label, "probability": round(float(probability), 4)}


app = FastAPI(title="Credit Card Default Predictor")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mlp_loaded": (MODELS_DIR / "mlp_model.pt").exists(),
        "tree_loaded": (MODELS_DIR / "catboost_model.joblib").exists(),
    }


@app.post("/predict/mlp")
def predict_mlp(payload: CreditDefaultRequest) -> dict:
    frame = payload_to_frame(payload)
    metadata, model = load_mlp_assets()
    cat_tensor, num_tensor = encode_for_mlp(frame, metadata)
    with torch.no_grad():
        probability = torch.sigmoid(model(cat_tensor, num_tensor)).item()
    return prediction_response(probability)


@app.post("/predict/tree")
def predict_tree(payload: CreditDefaultRequest) -> dict:
    frame = payload_to_frame(payload)
    frame = frame[ALL_FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        frame[col] = frame[col].astype(str)
    for col in NUMERICAL_COLUMNS:
        frame[col] = frame[col].astype(float)
    model = load_tree_model()
    pool = Pool(frame, cat_features=CATEGORICAL_COLUMNS)
    probability = model.predict_proba(pool)[:, 1][0]
    return prediction_response(probability)
