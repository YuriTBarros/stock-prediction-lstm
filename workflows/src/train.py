"""
train.py - Script de treinamento do modelo LSTM

Este script orquestra o pipeline de treinamento usando as funções
testadas do my_model_lib.py.
"""

import argparse
import joblib
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import json

from src.data_ingestion import download_stock_data
from src.my_model_lib import build_model, make_windowed_ds, set_global_seed


def calculate_metrics(y_true, y_pred):
    """Calcula métricas de avaliação"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0)
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


def inverse_transform(y, scaler):
    """Desnormaliza os valores"""
    return scaler.inverse_transform(y.reshape(-1, 1)).ravel()


def main():
    # Argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Treinar modelo LSTM para previsão de ações")
    parser.add_argument("--ticker", default="SPY", help="Código da ação")
    parser.add_argument("--period", default="2y", help="Período de dados")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--lookback", type=int, default=60, help="Janela de entrada")
    parser.add_argument("--horizon", type=int, default=1, help="Horizonte de predição")
    parser.add_argument("--units", type=int, default=128, help="Unidades LSTM")
    parser.add_argument("--dropout", type=float, default=0.2, help="Taxa de dropout")
    parser.add_argument("--arch", default="stacked", choices=["lstm", "stacked"], help="Arquitetura")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 TREINAMENTO DE MODELO LSTM")
    print("=" * 60)
    print(f"Ação: {args.ticker}")
    print(f"Período: {args.period}")
    print(f"Arquitetura: {args.arch}")
    print(f"Épocas: {args.epochs}")
    print(f"Lookback: {args.lookback} dias")
    print(f"Horizonte: {args.horizon} dia(s)")
    print("=" * 60)
    
    # Definir seed para reprodutibilidade
    set_global_seed(42)
    
    # 1. Carregar ou baixar dados
    print("\n📥 Etapa 1: Carregando dados")
    
    # Tentar usar dados locais primeiro (evita problemas com Yahoo Finance)
    local_file = Path(f"./data/{args.ticker.upper()}_data.parquet")
    
    if local_file.exists():
        print(f"✅ Usando dados locais: {local_file}")
        data_file = local_file
    else:
        print(f"⚠️  Dados locais não encontrados. Baixando do Yahoo Finance...")
        data_file = download_stock_data(args.ticker, args.period)
    
    df = pd.read_parquet(data_file)
    
    # Lidar com MultiIndex nas colunas (comum em dados do Yahoo Finance)
    if isinstance(df.columns, pd.MultiIndex):
        # Se for MultiIndex, pega apenas o primeiro nível (nome da coluna)
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        # Caso contrário, converte normalmente para minúsculas
        df.columns = [str(col).lower() for col in df.columns]

    # Garantir ordenação temporal
    df = df.sort_index()
    
    # 2. Definir features e target
    features = ["close", "high", "low", "open", "volume"]
    target_col = "close"
    
    print(f"\n📊 Etapa 2: Preparação dos dados")
    print(f"   Features: {features}")
    print(f"   Target: {target_col}")
    print(f"   Total de registros: {len(df)}")
    
    # 3. Split temporal (70% treino, 15% validação, 15% teste)
    n = len(df)
    train_size = int(n * 0.70)
    val_size = int(n * 0.15)
    
    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:train_size + val_size].copy()
    df_test = df.iloc[train_size + val_size:].copy()
    
    print(f"   Treino: {len(df_train)} registros")
    print(f"   Validação: {len(df_val)} registros")
    print(f"   Teste: {len(df_test)} registros")
    
    # 4. Criar scalers (fit apenas no treino)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_x.fit(df_train[features].astype("float32").values)
    scaler_y.fit(df_train[[target_col]].astype("float32").values)
    
    # 5. Criar janelas temporais (windowing)
    print(f"\n🔧 Etapa 3: Criação de janelas temporais")
    X_train, y_train = make_windowed_ds(df_train, features, target_col, args.lookback, args.horizon, scaler_x, scaler_y)
    X_val, y_val = make_windowed_ds(df_val, features, target_col, args.lookback, args.horizon, scaler_x, scaler_y)
    X_test, y_test = make_windowed_ds(df_test, features, target_col, args.lookback, args.horizon, scaler_x, scaler_y)
    
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val: {X_val.shape}")
    print(f"   X_test: {X_test.shape}")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("❌ Erro: Janelas insuficientes. Tente reduzir lookback ou aumentar período de dados.")
        return
    
    # 6. Criar modelo usando my_model_lib
    print(f"\n🏗️ Etapa 4: Construção do modelo")
    input_shape = (args.lookback, len(features))
    model = build_model(
        arch=args.arch,
        input_shape=input_shape,
        hidden=args.units,
        dropout=args.dropout,
        lr=0.001
    )
    
    print(f"   Arquitetura: {args.arch}")
    print(f"   Input shape: {input_shape}")
    model.summary()
    
    # 7. Configurar MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(f"{args.ticker.lower()}-lstm")
    
    # 8. Treinar com MLflow tracking
    print(f"\n🚀 Etapa 5: Treinamento")
    
    with mlflow.start_run():
        # Registrar parâmetros
        mlflow.log_params({
            "ticker": args.ticker,
            "period": args.period,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "lstm_units": args.units,
            "dropout": args.dropout,
            "arch": args.arch,
            "features": ",".join(features)
        })
        
        # Callbacks
        from tensorflow import keras
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Treinar
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 9. Avaliar no conjunto de teste
        print(f"\n📊 Etapa 6: Avaliação")
        
        y_test_pred_scaled = model.predict(X_test, verbose=0)
        y_test_true = inverse_transform(y_test.ravel(), scaler_y)
        y_test_pred = inverse_transform(y_test_pred_scaled.ravel(), scaler_y)
        
        metrics = calculate_metrics(y_test_true, y_test_pred)
        
        print(f"\n✅ Métricas no conjunto de teste:")
        print(f"   MAE:  {metrics['MAE']:.2f}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")
        
        # Registrar métricas no MLflow
        mlflow.log_metrics(metrics)
        
        # 10. Salvar artefatos
        print(f"\n💾 Etapa 7: Salvando artefatos")
        
        artifacts_dir = Path("./notebooks/artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Salvar modelo
        model_path = artifacts_dir / "best_model_final.keras"
        model.save(model_path)
        print(f"   Modelo salvo: {model_path}")
        
        # Salvar scalers
        scaler_x_path = artifacts_dir / "scaler_x_final.joblib"
        scaler_y_path = artifacts_dir / "scaler_y_final.joblib"
        joblib.dump(scaler_x, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        print(f"   Scalers salvos: {scaler_x_path}, {scaler_y_path}")
        
        # Salvar configuração
        config = {
            "ticker": args.ticker,
            "features": features,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "arch": args.arch,
            "metrics": metrics
        }
        
        config_path = artifacts_dir / "summary_final.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"   Config salvo: {config_path}")
        
        # Registrar artefatos no MLflow
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_x_path))
        mlflow.log_artifact(str(scaler_y_path))
        mlflow.log_artifact(str(config_path))
        
        print("\n" + "=" * 60)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 60)


if __name__ == "__main__":
    main()
