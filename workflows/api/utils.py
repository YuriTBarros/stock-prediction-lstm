from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json
import os
import numpy as np
import joblib

DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 5
DEFAULT_FEATURES = ["close"]

def load_summary(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

def build_feature_matrix(payload: List[Any], feature_columns: List[str]) -> np.ndarray:
    if len(payload) == 0:
        raise ValueError("Entrada vazia.")
    if isinstance(payload[0], (int, float)):
        if len(feature_columns) != 1:
            raise ValueError(
                "Entrada é 1-D, mas summary espera múltiplas features: %s" % feature_columns
            )
        return np.array(payload, dtype=float).reshape(-1, 1)
    if isinstance(payload[0], dict):
        rows = []
        for item in payload:
            row = []
            for col in feature_columns:
                if col not in item:
                    raise ValueError(f"Campo '{col}' ausente em {item}.")
                row.append(float(item[col]))
            rows.append(row)
        return np.array(rows, dtype=float)
    raise ValueError("Formato de entrada não suportado. Use lista de números ou de objetos.")

def make_autoregressive_forecast(model, scaler_x, scaler_y, history_features: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    X = ensure_2d(history_features)
    if X.shape[0] < lookback:
        pad_rows = lookback - X.shape[0]
        pad = np.repeat(X[:1, :], pad_rows, axis=0)
        X = np.vstack([pad, X])
    window = X[-lookback:, :]
    preds_scaled, preds_inverse = [], []
    for _ in range(horizon):
        w_scaled = scaler_x.transform(window)
        inp = np.expand_dims(w_scaled, axis=0)
        y_scaled = model.predict(inp, verbose=0)
        y_scaled_val = np.array(y_scaled).reshape(-1, 1)[0]
        y_inv = scaler_y.inverse_transform(y_scaled_val.reshape(1, -1))[0, 0]
        preds_scaled.append(y_scaled_val[0])
        preds_inverse.append(y_inv)
        next_row = window[-1, :].copy()
        next_row[0] = y_inv
        window = np.vstack([window[1:, :], next_row])
    return np.array(preds_scaled).reshape(-1, 1), np.array(preds_inverse).reshape(-1,)
