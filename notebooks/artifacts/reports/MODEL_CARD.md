# Model Card — LSTM Forecast (D+1)

## Visão Geral
- **Alvo:** close
- **Features:** close, high, low, open, volume
- **Lookback:** 60 dias
- **Arquitetura:** lstm (hidden=128, dropout=0.2)
- **Scaler:** minmax
- **Treino:** epochs=100, batch=64, lr=0.0005

## Partições
- **Treino:** 1229 linhas
- **Validação:** 264 linhas
- **Teste:** 264 linhas

## Métricas Finais (Teste)
{
  "MAE": 7.005677700042725,
  "RMSE": 10.020668033708974,
  "MAPE": 1.2126956135034561
}
