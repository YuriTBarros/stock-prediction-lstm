# my_model_lib.py
# -------------------------------------------------------------
# Funções utilitárias para o LSTM: build_model() e make_windowed_ds()
# Reprodutibilidade e compatibilidade com Apple Silicon.
# -------------------------------------------------------------

from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===== Reprodutibilidade (seeds) =====
def set_global_seed(seed: int = 42):
    import os, random
    os.environ["TF_DETERMINISTIC_OPS"] = "1"  # melhor reprodutibilidade (pode reduzir perf)
    np.random.seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

# ===== Janela deslizante (windowing) =====
def make_windowed_ds(
    df,
    features: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
    scaler_x,
    scaler_y,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constrói janelas temporais para LSTM.
    - df: DataFrame ordenado temporalmente.
    - features: colunas de entrada (na ordem do treino).
    - target_col: coluna alvo (close, p.ex.).
    - lookback: nº de passos históricos por amostra (ex.: 40).
    - horizon: nº de passos à frente a prever (ex.: 1).
    - scaler_x / scaler_y: scalers já ajustados (fit) no split correto.
    Retorna:
      X: shape (n_amostras, lookback, n_features)
      y: shape (n_amostras, 1)
    """
    X, y = [], []
    # dados originais (float32)
    feats_np = df[features].astype("float32").values
    tgt_np   = df[[target_col]].astype("float32").values

    # aplica os scalers
    feats_scaled = scaler_x.transform(feats_np)
    tgt_scaled   = scaler_y.transform(tgt_np)

    n = len(df)
    # janelas: [i : i+lookback] -> prevendo valor em i+lookback+(horizon-1)
    last_start = n - lookback - horizon + 1
    if last_start <= 0:
        return np.empty((0, lookback, len(features)), dtype="float32"), np.empty((0, 1), dtype="float32")

    for i in range(last_start):
        X.append(feats_scaled[i : i + lookback, :])
        # prevemos o passo imediatamente após a janela (horizon=1) ou mais à frente
        target_idx = i + lookback + (horizon - 1)
        y.append(tgt_scaled[target_idx, 0])

    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32").reshape(-1, 1)
    return X, y

# ===== Modelo LSTM =====
def build_model(
    arch: str,
    input_shape: Tuple[int, int],  # (lookback, n_features)
    hidden: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
) -> keras.Model:
    """
    Constrói o modelo Keras conforme 'arch':
      - "lstm"   : 1 camada LSTM + Dense final
      - "stacked": 2 camadas LSTM (return_sequences=True na 1ª) + Dense final
    Compila com MSE; métrica MAE (as métricas globais você calcula fora no pós-processamento).
    """
    lookback, n_features = input_shape

    inputs = keras.Input(shape=(lookback, n_features))
    if arch == "stacked":
        x = layers.LSTM(hidden, return_sequences=True)(inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(hidden)(x)
    else:  # "lstm" (default)
        x = layers.LSTM(hidden)(inputs)

    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=f"LSTM_{arch}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )
    return model
