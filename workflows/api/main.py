"""
main.py - API FastAPI para previsão de ações

API simples para servir predições do modelo LSTM usando código testado.
"""

import os
import sys
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from tensorflow import keras
import joblib

# Configurar path para importações
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar funções utilitárias
from api.utils import (
    load_summary,
    build_feature_matrix,
    make_autoregressive_forecast,
    DEFAULT_LOOKBACK,
    DEFAULT_HORIZON,
    DEFAULT_FEATURES
)

# Configurações
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./notebooks/artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "best_model_final.keras"
SCALER_X_PATH = ARTIFACTS_DIR / "scaler_x_final.joblib"
SCALER_Y_PATH = ARTIFACTS_DIR / "scaler_y_final.joblib"
SUMMARY_PATH = ARTIFACTS_DIR / "summary_final.json"

# Criar app
app = FastAPI(
    title="API de Previsão de Ações",
    description="API para prever preços de ações usando LSTM",
    version="1.0.0"
)


# Modelos de dados
class PredictRequest(BaseModel):
    """Requisição de predição"""
    values: Optional[List[float]] = Field(None, description="Histórico como lista de números")
    records: Optional[List[Dict[str, Any]]] = Field(None, description="Histórico como lista de objetos")
    lookback: Optional[int] = None
    horizon: Optional[int] = None


class TrainRequest(BaseModel):
    """Requisição de treinamento"""
    ticker: str = "SPY"
    period: str = "2y"
    epochs: int = 50
    batch_size: int = 32
    lookback: int = 60
    horizon: int = 1
    units: int = 128
    dropout: float = 0.2
    arch: str = "stacked"


# Carregar modelo na inicialização
@app.on_event("startup")
async def load_assets():
    """Carrega o modelo e scalers salvos"""
    try:
        if not MODEL_PATH.exists():
            print(f"⚠️  Modelo não encontrado em {MODEL_PATH}")
            print("   Use POST /train para treinar um modelo primeiro")
            app.state.model = None
            app.state.scaler_x = None
            app.state.scaler_y = None
            app.state.summary = {}
            return
        
        app.state.model = keras.models.load_model(MODEL_PATH)
        app.state.scaler_x = joblib.load(SCALER_X_PATH)
        app.state.scaler_y = joblib.load(SCALER_Y_PATH)
        app.state.summary = load_summary(str(SUMMARY_PATH))
        
        print("✅ Modelo carregado com sucesso!")
        print(f"   Ticker: {app.state.summary.get('ticker', 'N/A')}")
        print(f"   Lookback: {app.state.summary.get('lookback', 'N/A')}")
        print(f"   Features: {app.state.summary.get('features', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        app.state.model = None
        app.state.scaler_x = None
        app.state.scaler_y = None
        app.state.summary = {}


# Endpoints
@app.get("/")
def root():
    """Informações da API"""
    model_loaded = app.state.model is not None
    return {
        "message": "API de Previsão de Ações com LSTM",
        "version": "1.0.0",
        "model_loaded": model_loaded,
        "endpoints": {
            "health": "/health",
            "healthz": "/healthz",
            "predict": "/predict",
            "train": "/train",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    """Verifica se a API está funcionando"""
    model_loaded = app.state.model is not None
    return {
        "status": "healthy" if model_loaded else "no_model",
        "model_loaded": model_loaded
    }


@app.get("/healthz")
def healthz():
    """Health check simples (compatibilidade)"""
    return {"ok": True}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Faz predição de preços futuros usando código testado do original.
    
    Aceita dois formatos de entrada:
    - **values**: Lista de números (apenas preço de fechamento)
    - **records**: Lista de objetos com múltiplas features
    
    Exemplo com values:
    ```json
    {
        "values": [100, 102, 101, 103, 105],
        "horizon": 5
    }
    ```
    
    Exemplo com records:
    ```json
    {
        "records": [
            {"close": 100, "high": 102, "low": 99, "open": 100, "volume": 1000000},
            {"close": 102, "high": 104, "low": 101, "open": 102, "volume": 1100000}
        ],
        "horizon": 5
    }
    ```
    """
    try:
        # Verificar se modelo está carregado
        if app.state.model is None:
            raise HTTPException(
                status_code=503,
                detail="Modelo não carregado. Treine um modelo primeiro usando POST /train"
            )
        
        # Obter configurações do summary ou usar defaults
        summary = app.state.summary or {}
        feature_columns = summary.get("features", DEFAULT_FEATURES)
        lookback = req.lookback or summary.get("lookback", DEFAULT_LOOKBACK)
        horizon = req.horizon or summary.get("horizon", DEFAULT_HORIZON)
        
        # Construir matriz de features usando função testada
        if req.values is not None:
            X_hist = build_feature_matrix(req.values, feature_columns)
        elif req.records is not None:
            X_hist = build_feature_matrix(req.records, feature_columns)
        else:
            raise HTTPException(
                status_code=422,
                detail="Forneça 'values' ou 'records'. Features esperadas: " + str(feature_columns)
            )
        
        # Fazer predição autoregressiva usando função testada
        _, preds = make_autoregressive_forecast(
            model=app.state.model,
            scaler_x=app.state.scaler_x,
            scaler_y=app.state.scaler_y,
            history_features=X_hist,
            lookback=lookback,
            horizon=horizon,
        )
        
        return {
            "horizon": horizon,
            "lookback": lookback,
            "features": feature_columns,
            "predictions": preds.tolist()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/train")
def train(request: TrainRequest):
    """
    Inicia o treinamento de um novo modelo.
    
    Exemplo:
    ```json
    {
        "ticker": "PETR4.SA",
        "period": "2y",
        "epochs": 50,
        "arch": "stacked"
    }
    ```
    """
    # Construir comando de treinamento
    cmd = [
        "python3", "-m", "src.train",
        "--ticker", request.ticker,
        "--period", request.period,
        "--epochs", str(request.epochs),
        "--batch_size", str(request.batch_size),
        "--lookback", str(request.lookback),
        "--horizon", str(request.horizon),
        "--units", str(request.units),
        "--dropout", str(request.dropout),
        "--arch", request.arch
    ]
    
    try:
        print(f"🚀 Iniciando treinamento: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd="/app",
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos
        )
        
        if result.returncode == 0:
            # Recarregar modelo após treinamento
            try:
                app.state.model = keras.models.load_model(MODEL_PATH)
                app.state.scaler_x = joblib.load(SCALER_X_PATH)
                app.state.scaler_y = joblib.load(SCALER_Y_PATH)
                app.state.summary = load_summary(str(SUMMARY_PATH))
                print("✅ Modelo recarregado após treinamento")
            except Exception as e:
                print(f"⚠️  Erro ao recarregar modelo: {e}")
            
            return {
                "status": "success",
                "message": "Modelo treinado com sucesso!",
                "ticker": request.ticker,
                "output": result.stdout[-1000:]  # Últimos 1000 chars
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Erro no treinamento: {result.stderr[-1000:]}"
            )
    
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=500,
            detail="Treinamento excedeu o tempo limite de 10 minutos"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao executar treinamento: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
