"""
retrain.py — automatiza o re-treino LSTM com MLflow
---------------------------------------------------
Requisitos:
 - dados .parquet com colunas ['close','high','low','open','volume']
 - src/my_model_lib.py com build_model(), make_windowed_ds(), set_global_seed()
 - MLflow rodando (local ou remoto) ou file:./mlruns

Uso (a partir da RAIZ do projeto):
  python -m src.retrain \
    --ticker SPY \
    --data_path ./data/spy.parquet \
    --mlflow_uri file:./mlruns \
    --improve_delta 0.02
"""

import os, sys, math, json, argparse, joblib
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# --- IMPORT do pacote local src/ ---
from src.my_model_lib import build_model, make_windowed_ds, set_global_seed

# =========================
# Configuração de argumentos
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--ticker", default="SPY")
parser.add_argument("--data_path", required=True, help="Caminho para arquivo parquet")
parser.add_argument("--mlflow_uri", default="file:./mlruns")
parser.add_argument("--improve_delta", type=float, default=0.02, help="Ganho mínimo (proporcional) para promover novo modelo")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

TICKER        = str(args.ticker)
MLFLOW_URI    = str(args.mlflow_uri)
IMPROVE_DELTA = float(args.improve_delta)
SEED          = int(args.seed)

# =========================
# Pastas de artefatos
# =========================
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = (ROOT / "artifacts"); ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR   = ARTIFACTS_DIR / "reports"; REPORTS_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# Helpers locais
# =========================
def make_scaler(kind): 
    return MinMaxScaler() if kind == "minmax" else StandardScaler()

def inverse_transform(v, sc): 
    return sc.inverse_transform(v.reshape(-1,1)).ravel()

def metrics_report(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}

def ensure_sorted_index(df: pd.DataFrame) -> pd.DataFrame:
    """Garante ordenação temporal por index; tenta converter index pra datetime se possível."""
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.set_index(pd.to_datetime(df.index))
        except Exception:
            pass
    return df.sort_index()

def to_native(obj):
    import numpy as np, pandas as pd
    from pathlib import Path
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)): return obj.isoformat()
    if isinstance(obj, (pd.Series, pd.Index)): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_native(v) for v in obj]
    return obj

# =========================
# Inicializações
# =========================
set_global_seed(SEED)
mlflow.set_tracking_uri(MLFLOW_URI)

# =========================
# Carregar dados
# =========================
df = pd.read_parquet(args.data_path)
df = df.rename(columns=str.lower)
df = ensure_sorted_index(df)

# Checagens básicas
for c in ["close","high","low","open","volume"]:
    assert c in df.columns, f"Coluna obrigatória ausente: {c}"

TARGET_COL = "close"
FEATURES   = ["close","high","low","open","volume"]

# Split temporal 70/15/15
n = len(df)
train_size = int(n * 0.70)
val_size   = int(n * 0.15)
df_train   = df.iloc[:train_size].copy()
df_val     = df.iloc[train_size:train_size+val_size].copy()
df_test    = df.iloc[train_size+val_size:].copy()

print(f"[INFO] Shapes -> train={df_train.shape}, val={df_val.shape}, test={df_test.shape}")

# =========================
# STEP 9 — mini-grid (resumido)
# =========================
mlflow.set_experiment(f"{TICKER.lower()}-lstm-grid")
best = {"val_RMSE": float("inf")}

# Você pode expandir esta lista com mais cenários
SCENARIOS = [
    dict(features=FEATURES, lookback=40, arch="lstm",    lr=1e-3, hidden=128, dropout=0.2, scaler="minmax", horizon=1, epochs=100, batch=64),
    # dict(features=FEATURES, lookback=60, arch="stacked", lr=5e-4, hidden=128, dropout=0.2, scaler="minmax", horizon=1, epochs=120, batch=64),
]

for i, cfg in enumerate(SCENARIOS, 1):
    with mlflow.start_run(run_name=f"scenario_{i}"):
        print(f"[GRID] Cenário {i}/{len(SCENARIOS)} -> {cfg}")
        # Fit scalers no TREINO
        sx, sy = make_scaler(cfg["scaler"]), make_scaler(cfg["scaler"])
        sx.fit(df_train[cfg["features"]].astype("float32").values)
        sy.fit(df_train[[TARGET_COL]].astype("float32").values)

        # Windowing
        X_tr, y_tr = make_windowed_ds(df_train, cfg["features"], TARGET_COL, cfg["lookback"], cfg["horizon"], sx, sy)
        X_va, y_va = make_windowed_ds(df_val,   cfg["features"], TARGET_COL, cfg["lookback"], cfg["horizon"], sx, sy)
        X_te, y_te = make_windowed_ds(df_test,  cfg["features"], TARGET_COL, cfg["lookback"], cfg["horizon"], sx, sy)
        if len(X_tr) == 0 or len(X_va) == 0 or len(X_te) == 0:
            mlflow.set_tag("status", "skipped_window_too_short")
            print("[WARN] Janelas insuficientes; cenário pulado.")
            continue

        # Modelo
        model = build_model(cfg["arch"], (cfg["lookback"], len(cfg["features"])), int(cfg["hidden"]), float(cfg["dropout"]), float(cfg["lr"]))

        ckpt = ARTIFACTS_DIR / "tmp.keras"
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        ]

        model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=int(cfg["epochs"]), batch_size=int(cfg["batch"]), verbose=0, callbacks=callbacks)
        model = keras.models.load_model(ckpt)

        # Avaliação (VAL para seleção; TEST só informativo)
        y_va_pred = inverse_transform(model.predict(X_va, verbose=0).ravel(), sy)
        y_va_true = inverse_transform(y_va.ravel(), sy)
        val_m = metrics_report(y_va_true, y_va_pred)

        y_te_pred = inverse_transform(model.predict(X_te, verbose=0).ravel(), sy)
        y_te_true = inverse_transform(y_te.ravel(), sy)
        test_m = metrics_report(y_te_true, y_te_pred)

        test_m = {k: float(v) for k, v in test_m.items()}


        # Log no MLflow (sempre tipos nativos)
        mlflow.log_params({
            "features": ",".join(cfg["features"]),
            "lookback": int(cfg["lookback"]),
            "arch": str(cfg["arch"]),
            "lr": float(cfg["lr"]),
            "hidden": int(cfg["hidden"]),
            "dropout": float(cfg["dropout"]),
            "scaler": str(cfg["scaler"]),
            "horizon": int(cfg["horizon"]),
            "epochs": int(cfg["epochs"]),
            "batch": int(cfg["batch"]),
            "target": TARGET_COL
        })
        mlflow.log_metrics({
            "val_MAE": float(val_m["MAE"]), "val_RMSE": float(val_m["RMSE"]), "val_MAPE": float(val_m["MAPE"]),
            "test_MAE": float(test_m["MAE"]), "test_RMSE": float(test_m["RMSE"]), "test_MAPE": float(test_m["MAPE"])
        })
        mlflow.set_tags({"selection_criterion":"val_RMSE","ticker":TICKER,"environment":"script-retrain"})

        if val_m["RMSE"] < best["val_RMSE"]:
            best = {
                **cfg,
                "val_RMSE": float(val_m["RMSE"]),
                "val_MAE": float(val_m["MAE"]),
                "val_MAPE": float(val_m["MAPE"]),
                "mlflow_run_id": mlflow.active_run().info.run_id
            }
            # salva artefatos locais do melhor do grid
            model.save(ARTIFACTS_DIR / "best_model.keras")
            joblib.dump(sx, ARTIFACTS_DIR / "scaler_x.joblib")
            joblib.dump(sy, ARTIFACTS_DIR / "scaler_y.joblib")

# =========================
# STEP 10 — teste final / re-treino
# =========================
assert best.get("mlflow_run_id"), "Nenhum cenário válido no grid. Ajuste seu SCENARIOS/Splits."

mlflow.set_experiment(f"{TICKER.lower()}-lstm-final")

# Normaliza tipos em best_cfg
best_cfg = {
    "features": list(best["features"]),
    "lookback": int(best["lookback"]),
    "arch":     str(best["arch"]),
    "lr":       float(best["lr"]),
    "hidden":   int(best["hidden"]),
    "dropout":  float(best["dropout"]),
    "scaler":   str(best["scaler"]),
    "horizon":  int(best["horizon"]),
    "epochs":   int(best["epochs"]),
    "batch":    int(best["batch"]),
}


df_trval = pd.concat([df_train, df_val]).copy()

# 10% do Train+Val vira mini-val interna para EarlyStopping
split = max(int(len(df_trval) * 0.90), best_cfg["lookback"] + best_cfg["horizon"] + 1)
df_tr_inner = df_trval.iloc[:split]
df_vi_inner = df_trval.iloc[split:]

sx = make_scaler(best_cfg["scaler"])
sy = make_scaler(best_cfg["scaler"])
sx.fit(df_tr_inner[best_cfg["features"]].astype("float32").values)
sy.fit(df_tr_inner[[TARGET_COL]].astype("float32").values)

X_tr, y_tr = make_windowed_ds(df_tr_inner, best_cfg["features"], TARGET_COL, best_cfg["lookback"], best_cfg["horizon"], sx, sy)
X_vi, y_vi = make_windowed_ds(df_vi_inner, best_cfg["features"], TARGET_COL, best_cfg["lookback"], best_cfg["horizon"], sx, sy)
X_te, y_te = make_windowed_ds(df_test,     best_cfg["features"], TARGET_COL, best_cfg["lookback"], best_cfg["horizon"], sx, sy)

with mlflow.start_run(run_name="final_retrain_test"):
    final_model = build_model(best_cfg["arch"], (best_cfg["lookback"], len(best_cfg["features"])), best_cfg["hidden"], best_cfg["dropout"], best_cfg["lr"])

    ckpt_final = ARTIFACTS_DIR / "final_tmp.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(ckpt_final), monitor="val_loss", save_best_only=True),
    ]

    final_model.fit(X_tr, y_tr, validation_data=(X_vi, y_vi), epochs=best_cfg["epochs"], batch_size=best_cfg["batch"], verbose=0, callbacks=callbacks)
    final_model = keras.models.load_model(ckpt_final)

    y_te_pred_scaled = final_model.predict(X_te, verbose=0).ravel()
    y_te_true = inverse_transform(y_te.ravel(), sy)
    y_te_pred = inverse_transform(y_te_pred_scaled, sy)
    test_m = metrics_report(y_te_true, y_te_pred)
    print("\n=== MÉTRICAS FINAIS (TEST) ===", test_m)

    mlflow.log_params({
        "features": ",".join(best_cfg["features"]),
        **{k: best_cfg[k] for k in ["lookback","arch","lr","hidden","dropout","scaler","horizon","epochs","batch"]},
        "target": TARGET_COL,
        "grid_best_run_id": best.get("mlflow_run_id", "n/a")
    })
    mlflow.log_metrics({f"test_final_{k}": float(v) for k, v in test_m.items()})

    # Gráficos simples
    plt.figure(figsize=(10,4))
    plt.plot(y_te_true, label="Real")
    plt.plot(y_te_pred, label="Previsto (final)")
    plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS_DIR / "test_real_vs_pred.png"); plt.close()

    errors = y_te_true - y_te_pred
    plt.figure(figsize=(10,3))
    plt.plot(errors); plt.axhline(0, color='red', ls='--')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "test_residuos.png"); plt.close()

    # Artefatos finais
    final_model_path = ARTIFACTS_DIR / "best_model_final.keras"
    final_sx_path    = ARTIFACTS_DIR / "scaler_x_final.joblib"
    final_sy_path    = ARTIFACTS_DIR / "scaler_y_final.joblib"
    final_model.save(final_model_path)
    joblib.dump(sx, final_sx_path)
    joblib.dump(sy, final_sy_path)

    summary_final = {
        "schema_version": 1,
        "ticker": TICKER,
        "target": TARGET_COL,
        **best_cfg,
        "test_final": test_m,
        "model_final":     str(final_model_path),
        "scaler_x_final":  str(final_sx_path),
        "scaler_y_final":  str(final_sy_path),
        "reason": "Menor val_RMSE no grid; re-treino em Train+Val; teste final único."
    }
    with open(ARTIFACTS_DIR / "summary_final.json", "w") as f:
        json.dump(to_native(summary_final), f, indent=2)

    # Log no MLflow
    for p in [final_model_path, final_sx_path, final_sy_path,
              ARTIFACTS_DIR / "summary_final.json",
              REPORTS_DIR / "test_real_vs_pred.png",
              REPORTS_DIR / "test_residuos.png"]:
        mlflow.log_artifact(str(p))

    # ===== Registrar no Model Registry (com fallback) =====
    try:
        sample_input = np.zeros((1, best_cfg["lookback"], len(best_cfg["features"])), dtype=np.float32)
        signature = infer_signature(sample_input, np.array([[0.0]], dtype=np.float32))

        mlflow.keras.log_model(
            model=final_model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"LSTM-{TICKER}"
        )
        mlflow.set_tag("registry", "registered")
    except Exception as e:
        print(f"[INFO] Registry indisponível. Logando apenas como artefato do run. Detalhe: {e}")
        mlflow.keras.log_model(model=final_model, artifact_path="model_unregistered")
        mlflow.set_tag("registry", "not_available")

# =========================
# Comparação com Production e promoção por métrica

# =========================
# Comparação com Production e promoção por métrica
# =========================
client = MlflowClient()
MODEL_NAME = f"LSTM-{TICKER}"
ALIAS_NAME = "Production"

def get_test_rmse_from_run(run_id: str) -> float:
    try:
        run = client.get_run(run_id)
        return float(run.data.metrics.get("test_final_RMSE"))
    except Exception:
        return float("inf")

def get_latest_version():
    try:
        versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
        if not versions:
            return None
        return sorted(versions, key=lambda mv: int(mv.version))[-1]
    except Exception:
        return None

def get_prod_version_with_alias():
    # preferir alias se a versão do MLflow suportar
    if hasattr(client, "get_model_version_by_alias"):
        try:
            return client.get_model_version_by_alias(MODEL_NAME, ALIAS_NAME)
        except Exception:
            return None
    return None

def get_prod_version_with_stage():
    try:
        versions = list(client.search_model_versions(f"name='{MODEL_NAME}'"))
        pros = [mv for mv in versions if getattr(mv, "current_stage", "") == "Production"]
        return pros[0] if pros else None
    except Exception:
        return None

def set_prod_alias_or_stage(new_version: int):
    """Tenta usar alias 'Production'; se indisponível, usa estágio Production."""
    if hasattr(client, "set_registered_model_alias"):
        # remove alias anterior (opcional), depois seta no novo
        if hasattr(client, "delete_registered_model_alias"):
            try:
                client.delete_registered_model_alias(MODEL_NAME, ALIAS_NAME)
            except Exception:
                pass
        client.set_registered_model_alias(MODEL_NAME, ALIAS_NAME, new_version)
        print(f"[OK] Alias '{ALIAS_NAME}' apontando para v{new_version}.")
    else:
        # fallback: usar stage Production (arquiva anteriores)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"[OK] Versão v{new_version} marcada como stage=Production.")

newv = get_latest_version()
if newv is None:
    print("[INFO] Model Registry não disponível ou sem versões. Promoção ignorada.")
    sys.exit(0)

# Descobrir 'Production' atual (por alias ou por stage)
prod = get_prod_version_with_alias() or get_prod_version_with_stage()

# Se nunca houve Production, promova direto
if prod is None:
    set_prod_alias_or_stage(int(newv.version))
    print(f"[OK] Sem Production anterior. Promovido {TICKER} v{newv.version} para Production.")
    sys.exit(0)

cur_rmse = get_test_rmse_from_run(prod.run_id)
new_rmse = get_test_rmse_from_run(newv.run_id)

# Regra: promove se melhora relativa >= IMPROVE_DELTA
if (cur_rmse - new_rmse) / max(cur_rmse, 1e-9) >= IMPROVE_DELTA:
    set_prod_alias_or_stage(int(newv.version))
    print(f"[OK] Promovido {TICKER} v{newv.version} para Production (RMSE {new_rmse:.4f} < {cur_rmse:.4f}).")
else:
    print(f"[INFO] Não promoveu: nova RMSE {new_rmse:.4f} não melhora >= {IMPROVE_DELTA*100:.1f}% sobre {cur_rmse:.4f}.")

