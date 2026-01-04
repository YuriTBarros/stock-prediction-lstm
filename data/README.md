# 📊 Dados Históricos Incluídos

Este diretório contém dados históricos de ações já baixados, permitindo que o projeto funcione **sem depender do Yahoo Finance**.

## Arquivos Disponíveis

| Arquivo | Ticker | Período | Registros | Última Atualização |
| :--- | :--- | :--- | ---: | :--- |
| `SPY_data.parquet` | SPY (S&P 500 ETF) | ~7 anos | 1757 | 15/10/2025 |

## Como Usar

O script `src/train.py` **automaticamente usa dados locais** se disponíveis:

```bash
# Treinar com dados locais (SPY)
python -m src.train --ticker SPY --epochs 50

# Se o arquivo existir, não precisa de internet!
```

## Adicionar Novos Dados

Para adicionar dados de outros tickers:

```bash
# Baixar dados manualmente
python -m src.data_ingestion AAPL 2y

# Ou via API
curl -X POST http://localhost:8000/train \
  -d '{"ticker": "AAPL", "period": "2y", "epochs": 50}'
```

## Formato dos Dados

Os arquivos `.parquet` contêm:

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `close` | float64 | Preço de fechamento ajustado |
| `high` | float64 | Preço máximo do dia |
| `low` | float64 | Preço mínimo do dia |
| `open` | float64 | Preço de abertura |
| `volume` | int64 | Volume negociado |

**Índice:** `Date` (DatetimeIndex)

## Vantagens dos Dados Locais

✅ **Funciona offline** - Não precisa de internet  
✅ **Mais rápido** - Não precisa baixar  
✅ **Confiável** - Não depende da disponibilidade do Yahoo Finance  
✅ **Reprodutível** - Sempre os mesmos dados  

## Atualizar Dados

Para atualizar os dados do SPY:

```bash
# Forçar download mesmo com dados locais
rm data/SPY_data.parquet
python -m src.data_ingestion SPY 2y
```

---

**Nota:** Os dados são baixados do Yahoo Finance e salvos localmente para uso futuro.
