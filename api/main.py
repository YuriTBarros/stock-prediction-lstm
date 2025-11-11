from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import numpy as np
import joblib
from keras.models import load_model
from .utils import load_summary, build_feature_matrix, make_autoregressive_forecast, DEFAULT_LOOKBACK, DEFAULT_HORIZON, DEFAULT_FEATURES

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./notebooks/artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model_final.keras")
SCALER_X_PATH = os.path.join(ARTIFACTS_DIR, "scaler_x_final.joblib")
SCALER_Y_PATH = os.path.join(ARTIFACTS_DIR, "scaler_y_final.joblib")
SUMMARY_PATH = os.path.join(ARTIFACTS_DIR, "summary_final.json")

app = FastAPI(title="Stock Forecast API (Keras + FastAPI)")

class PredictRequest(BaseModel):
    values: Optional[List[float]] = Field(None, description="Histórico como lista de números")
    records: Optional[List[Dict[str, Any]]] = Field(None, description="Histórico como lista de objetos")
    lookback: Optional[int] = None
    horizon: Optional[int] = None

@app.on_event("startup")
def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo não encontrado em {MODEL_PATH}")
    app.state.model = load_model(MODEL_PATH)
    if not os.path.exists(SCALER_X_PATH) or not os.path.exists(SCALER_Y_PATH):
        raise RuntimeError("Scalers não encontrados.")
    app.state.scaler_x = joblib.load(SCALER_X_PATH)
    app.state.scaler_y = joblib.load(SCALER_Y_PATH)
    app.state.summary = load_summary(SUMMARY_PATH)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        summary = app.state.summary or {}
        feature_columns = summary.get("features", DEFAULT_FEATURES)
        lookback = req.lookback or summary.get("lookback", DEFAULT_LOOKBACK)
        horizon = req.horizon or summary.get("horizon", DEFAULT_HORIZON)
        if req.values is not None:
            X_hist = build_feature_matrix(req.values, feature_columns)
        elif req.records is not None:
            X_hist = build_feature_matrix(req.records, feature_columns)
        else:
            raise HTTPException(status_code=422, detail="Forneça 'values' ou 'records'")
        _, preds = make_autoregressive_forecast(
            model=app.state.model,
            scaler_x=app.state.scaler_x,
            scaler_y=app.state.scaler_y,
            history_features=X_hist,
            lookback=lookback,
            horizon=horizon,
        )
        return {"horizon": horizon, "lookback": lookback, "features": feature_columns, "predictions": preds.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "OK. Use POST /predict"}
