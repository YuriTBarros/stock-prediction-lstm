# Model Card — LSTM Forecast (D+1)

## Visão Geral
- **Alvo:** close
- **Features:** close, high, low, open, volume
- **Lookback:** 40 dias
- **Arquitetura:** lstm (hidden=128, dropout=0.2)
- **Scaler:** minmax
- **Treino:** epochs=100, batch=64, lr=0.001

## Partições
- **Treino:** 1229 linhas
- **Validação:** 264 linhas
- **Teste:** 264 linhas

## Métricas Finais (Teste)
{
  "MAE": 6.439180850982666,
  "RMSE": 9.505464688680608,
  "MAPE": 1.11744599416852
}
