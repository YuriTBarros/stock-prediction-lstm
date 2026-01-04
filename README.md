# Stock Price Prediction com LSTM e MLOps

**Projeto de Pós-Graduação em Machine Learning**

Sistema completo de previsão de preços de ações utilizando redes neurais LSTM (Long Short-Term Memory) com pipeline MLOps automatizado para experimentação, treinamento, deployment e monitoramento.

---

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Arquitetura](#arquitetura)
3. [Tecnologias](#tecnologias)
4. [Pré-requisitos](#pré-requisitos)
5. [Instalação](#instalação)
6. [Execução](#execução)
7. [Uso da API](#uso-da-api)
8. [Workflows Automatizados](#workflows-automatizados)
9. [Notebooks](#notebooks)
10. [Estrutura do Projeto](#estrutura-do-projeto)
11. [Métricas do Modelo](#métricas-do-modelo)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

Este projeto implementa um pipeline MLOps completo para previsão de séries temporais financeiras, demonstrando a aplicação prática de conceitos modernos de Machine Learning Operations em um problema real.

### Características Principais

- **Modelo LSTM** treinado com 7 anos de dados históricos do SPY (S&P 500 ETF)
- **3 arquiteturas** testadas: LSTM simples, Stacked LSTM e BiLSTM
- **Grid Search** automatizado com 32 combinações de hiperparâmetros
- **API REST** com FastAPI para predições e treinamento
- **Rastreamento de experimentos** com MLflow (58 experimentos históricos)
- **Orquestração de workflows** com n8n para automação
- **Notificações em tempo real** via Discord
- **Containerização completa** com Docker Compose
- **Modelo pré-treinado** pronto para uso (MAPE: 1.21%)

### Objetivos do Projeto

1. Demonstrar implementação de pipeline MLOps de ponta a ponta
2. Automatizar ciclo de vida de modelos de Machine Learning
3. Garantir reprodutibilidade e rastreabilidade de experimentos
4. Facilitar deployment e monitoramento de modelos em produção

---

## 🏗️ Arquitetura

O sistema é composto por 6 serviços containerizados que trabalham em conjunto:

![Arquitetura do Sistema](images/architecture.png)

### Componentes

| Componente | Tecnologia | Porta | Função |
| :--- | :--- | :---: | :--- |
| **API** | FastAPI | 8000 | Endpoints de predição e treinamento |
| **MLflow** | MLflow Server | 5000 | Rastreamento de experimentos e model registry |
| **n8n** | n8n Workflow | 5678 | Orquestração de workflows automatizados |
| **PostgreSQL** | PostgreSQL 15 | 5432 | Backend para MLflow e n8n |
| **Discord Proxy** | Flask | 9094 | Proxy para notificações Discord |
| **Prometheus** | Prometheus | 9090 | Monitoramento de métricas (opcional) |

### Fluxo de Dados

```
┌─────────────┐
│   n8n       │ ──── Agenda treinamento/predição
└─────────────┘
       │
       ▼
┌─────────────┐
│   API       │ ──── Processa requisições
└─────────────┘
       │
       ├──────▶ Yahoo Finance (dados)
       │
       ├──────▶ MLflow (tracking)
       │
       └──────▶ Discord (notificações)
```

---

## 🛠️ Tecnologias

### Machine Learning
- **TensorFlow/Keras** - Framework de deep learning
- **scikit-learn** - Pré-processamento e métricas
- **pandas/numpy** - Manipulação de dados

### MLOps
- **MLflow** - Experiment tracking e model registry
- **n8n** - Workflow automation
- **Docker/Docker Compose** - Containerização

### API e Backend
- **FastAPI** - Framework web moderno
- **PostgreSQL** - Banco de dados relacional
- **yfinance** - Dados financeiros do Yahoo Finance

### Monitoramento
- **Discord Webhooks** - Notificações em tempo real
- **Prometheus** - Coleta de métricas (opcional)

---

## 📋 Pré-requisitos

### Software Necessário

- **Docker** (versão 20.10 ou superior)
- **Docker Compose** (versão 2.0 ou superior)
- **Git** (para clonar o repositório)

### Opcional (para desenvolvimento)

- **Python 3.11+** (para executar notebooks)
- **Jupyter Notebook** (para explorar notebooks originais)

### Verificar Instalação

```bash
# Verificar Docker
docker --version
# Saída esperada: Docker version 20.10.x ou superior

# Verificar Docker Compose
docker compose version
# Saída esperada: Docker Compose version v2.x.x ou superior
```

---

## 🚀 Instalação

### Passo 1: Obter o Projeto

```bash
# Extrair o arquivo ZIP
unzip stock-prediction-mlops.zip
cd stock-prediction
```

### Passo 2: Configurar Variáveis de Ambiente (Opcional)

Para receber notificações no Discord, crie um arquivo `.env`:

```bash
# Criar arquivo .env
cat > .env << EOF
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/SEU_WEBHOOK_AQUI
PORT=8000
EOF
```

**Como obter o webhook do Discord:**
1. Acesse seu servidor Discord
2. Vá em Configurações do Canal → Integrações → Webhooks
3. Clique em "Novo Webhook"
4. Copie a URL do webhook

### Passo 3: Verificar Estrutura

```bash
# Verificar arquivos principais
ls -l docker-compose.yml
ls -l notebooks/*.ipynb
ls -l notebooks/artifacts/best_model_final.keras
```

---

## ▶️ Execução

### Iniciar Todos os Serviços

```bash
# Iniciar em background
docker compose up -d

# Aguardar inicialização (30-60 segundos)
sleep 30

# Verificar status dos serviços
docker compose ps
```

**Saída esperada:**
```
NAME                    STATUS    PORTS
stock-forecast-api      Up        0.0.0.0:8000->8000/tcp
mlops-mlflow            Up        0.0.0.0:5000->5000/tcp
mlops-n8n               Up        0.0.0.0:5678->5678/tcp
mlops-postgres          Up        0.0.0.0:5432->5432/tcp
discord-webhook-proxy   Up        0.0.0.0:9094->9094/tcp
```

### Acessar Interfaces Web

| Serviço | URL | Descrição |
| :--- | :--- | :--- |
| **API (Swagger)** | http://localhost:8000/docs | Documentação interativa da API |
| **MLflow UI** | http://localhost:5000 | Interface de experimentos |
| **n8n** | http://localhost:5678 | Editor de workflows |

### Verificar Saúde da API

```bash
# Testar endpoint de health
curl http://localhost:8000/health

# Resposta esperada:
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "/app/notebooks/artifacts/best_model_final.keras"
}
```

### Parar os Serviços

```bash
# Parar todos os serviços
docker compose down

# Parar e remover volumes (cuidado: apaga dados)
docker compose down -v
```

---

## 🔌 Uso da API

A API oferece endpoints para predição e treinamento de modelos.

### 1. Fazer Predição

Use o modelo pré-treinado para fazer predições:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "values": [100,102,101,103,105,104,106,108,107,109,
                111,110,112,114,113,115,117,116,118,120,
                119,121,123,122,124,126,125,127,129,128,
                130,132,131,133,135,134,136,138,137,139,
                141,140,142,144,143,145,147,146,148,150,
                149,151,153,152,154,156,155,157,159,158],
    "horizon": 5
  }'
```

**Resposta:**
```json
{
  "predictions": [160.23, 161.45, 162.18, 163.02, 163.89],
  "model_info": {
    "lookback": 60,
    "features": ["close", "high", "low", "open", "volume"]
  }
}
```

### 2. Treinar Novo Modelo

Treinar um modelo com configuração personalizada:

```bash
curl -X POST http://localhost:8000/train/single \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SPY",
    "epochs": 50,
    "lookback": 60,
    "arch": "lstm",
    "hidden": 128,
    "dropout": 0.2
  }'
```

**Parâmetros disponíveis:**
- `ticker`: Código da ação (ex: "SPY", "AAPL", "PETR4.SA")
- `epochs`: Número de épocas de treinamento (padrão: 50)
- `lookback`: Janela temporal de entrada (padrão: 60)
- `arch`: Arquitetura ("lstm", "stacked", "bilstm")
- `hidden`: Unidades LSTM (padrão: 128)
- `dropout`: Taxa de dropout (padrão: 0.2)

### 3. Grid Search

Executar busca de hiperparâmetros:

```bash
curl -X POST http://localhost:8000/train/grid \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "SPY"
  }'
```

Testa 32 combinações:
- Lookback: 20, 40, 60
- Arquitetura: lstm, stacked, bilstm
- Hidden units: 64, 128
- Dropout: 0.2, 0.3

### 4. Verificar Status

```bash
# Health check
curl http://localhost:8000/health

# Informações do modelo
curl http://localhost:8000/model/info
```

---

## 🔄 Workflows Automatizados

O projeto inclui 5 workflows n8n para automação completa do ciclo de vida do modelo.

### Workflows Disponíveis

| Workflow | Frequência | Função |
| :--- | :--- | :--- |
| **Grid Search** | Semanal (domingo 2h) | Busca de hiperparâmetros |
| **Treino Rápido** | Diário (6h) | Treinamento com 20 épocas |
| **Treino Automatizado** | Sob demanda | Treinamento completo |
| **Predição Diária** | Diário (9h) | Predições automáticas |
| **Monitoramento** | A cada 5 min | Health check da API |

### Configurar Workflows

1. **Acessar n8n:**
   ```
   http://localhost:5678
   ```

2. **Importar workflows:**
   - Clique em "Import from File"
   - Selecione arquivos da pasta `workflows/`
   - Importe os 5 arquivos JSON

3. **Ativar workflows:**
   - Abra cada workflow
   - Clique no toggle "Active" no canto superior direito

### Testar Workflow Manualmente

1. Abra o workflow desejado
2. Clique em "Execute Workflow"
3. Acompanhe a execução em tempo real
4. Verifique notificações no Discord (se configurado)

### Estrutura de Notificações

Cada workflow envia 3 tipos de notificações:

- **🚀 Início** - Workflow iniciado
- **✅ Sucesso** - Execução concluída com sucesso
- **❌ Erro** - Falha na execução com detalhes

---

## 📓 Notebooks

O projeto inclui 2 notebooks Jupyter que documentam todo o processo de desenvolvimento.

### Notebooks Disponíveis

#### 1. data_exploration.ipynb

Análise exploratória dos dados:
- Carregamento de dados do Yahoo Finance
- Estatísticas descritivas
- Visualizações de séries temporais
- Verificação de qualidade (valores nulos, outliers)
- Tratamento de MultiIndex

#### 2. lstm_model.ipynb

Desenvolvimento completo do modelo:
- Construção de arquiteturas LSTM
- Grid search de hiperparâmetros
- Treinamento com callbacks (EarlyStopping, ReduceLROnPlateau)
- Avaliação em conjunto de teste
- Análise de resíduos
- Salvamento de artefatos

### Executar Notebooks

```bash
# Instalar Jupyter (se necessário)
pip install jupyter notebook pandas matplotlib

# Navegar até a pasta
cd notebooks

# Iniciar Jupyter
jupyter notebook

# Abrir no navegador:
# http://localhost:8888
```

### Artifacts Gerados

Os notebooks geram artefatos na pasta `notebooks/artifacts/`:

- `best_model_final.keras` - Modelo treinado
- `scaler_x_final.joblib` - Scaler de features
- `scaler_y_final.joblib` - Scaler do target
- `summary_final.json` - Configuração do modelo
- `reports/` - Visualizações e resultados

---

## 📂 Estrutura do Projeto

```
stock-prediction/
│
├── README.md                    # Este arquivo
│
├── notebooks/                   # Jupyter Notebooks originais
│   ├── data_exploration.ipynb   # Análise exploratória
│   ├── lstm_model.ipynb         # Desenvolvimento do modelo
│   └── artifacts/               # Artefatos gerados
│       ├── best_model_final.keras
│       ├── scaler_x_final.joblib
│       ├── scaler_y_final.joblib
│       ├── summary_final.json
│       └── reports/             # Visualizações e CSVs
│
├── api/                         # API FastAPI
│   ├── main.py                  # Endpoints da API
│   └── utils.py                 # Funções utilitárias
│
├── src/                         # Código fonte
│   ├── data_ingestion.py        # Download de dados
│   ├── my_model_lib.py          # Arquiteturas LSTM
│   └── train.py                 # Pipeline de treinamento
│
├── workflows/                   # Workflows n8n
│   ├── grid_search.json
│   ├── treino_rapido.json
│   ├── treino_automatizado.json
│   ├── predicao_diaria.json
│   └── monitoramento.json
│
├── data/                        # Dados locais
│   └── SPY_data.parquet         # 7 anos de dados históricos
│
├── mlruns/                      # Histórico MLflow
│   └── [58 experimentos]
│
├── mlartifacts/                 # Artefatos MLflow
│
├── dockerfiles/                 # Dockerfiles
│   ├── Dockerfile               # API
│   ├── Dockerfile.mlflow        # MLflow
│   └── Dockerfile.postgres      # PostgreSQL
│
├── discord-webhook-proxy/       # Proxy Discord
│   └── app.py
│
├── docker-compose.yml           # Orquestração
├── requirements.txt             # Dependências Python
└── .env                         # Variáveis de ambiente (criar)
```

---

## 📊 Métricas do Modelo

### Modelo Pré-Treinado

O modelo incluído foi treinado com a seguinte configuração:

| Parâmetro | Valor |
| :--- | :--- |
| **Arquitetura** | LSTM Simples |
| **Lookback** | 60 dias |
| **Hidden Units** | 128 |
| **Dropout** | 0.2 |
| **Épocas** | 50 |
| **Batch Size** | 32 |
| **Learning Rate** | 0.001 |

### Resultados no Conjunto de Teste

| Métrica | Valor | Descrição |
| :--- | ---: | :--- |
| **MAE** | 7.01 | Erro absoluto médio |
| **RMSE** | 10.02 | Raiz do erro quadrático médio |
| **MAPE** | 1.21% | Erro percentual absoluto médio |

### Histórico de Experimentos

O projeto inclui **58 experimentos** rastreados no MLflow:

- **Experimento 1:** `lstm-spy-grid` (~55 runs)
  - Grid search com múltiplas configurações
  - Variações de lookback, arquitetura, learning rate, hidden units, dropout

- **Experimento 2:** `lstm-spy-final` (~3 runs)
  - Treinamento final com melhor configuração
  - Modelo com melhor RMSE selecionado

### Visualizar no MLflow

```bash
# Acessar MLflow UI
http://localhost:5000

# Navegar para:
# - "Experiments" para ver todos os experimentos
# - Selecionar múltiplos runs para comparar
# - Ver gráficos de métricas e parâmetros
```

---

## 🔧 Troubleshooting

### Problema: API não inicia

**Sintomas:**
```
Error: Cannot connect to API at localhost:8000
```

**Soluções:**

1. Verificar se o container está rodando:
```bash
docker compose ps
docker compose logs api
```

2. Verificar porta ocupada:
```bash
# Linux/Mac
lsof -i :8000

# Windows
netstat -ano | findstr :8000
```

3. Reiniciar serviço:
```bash
docker compose restart api
```

### Problema: Modelo não carregado

**Sintomas:**
```json
{
  "status": "healthy",
  "model_loaded": false
}
```

**Soluções:**

1. Verificar se o modelo existe:
```bash
ls -lh notebooks/artifacts/best_model_final.keras
```

2. Treinar novo modelo:
```bash
curl -X POST http://localhost:8000/train/single \
  -H "Content-Type: application/json" \
  -d '{"ticker": "SPY", "epochs": 50}'
```

### Problema: Yahoo Finance não responde

**Sintomas:**
```
Failed to get ticker 'SPY' reason: JSONDecodeError
```

**Soluções:**

1. **Use o modelo pré-treinado** (não precisa treinar):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [100,102,...,158], "horizon": 5}'
```

2. **Aguarde e tente novamente** (Yahoo Finance pode estar temporariamente indisponível)

3. **Use dados locais** (já incluídos):
```bash
# Os dados locais em data/SPY_data.parquet são usados automaticamente
# se o Yahoo Finance falhar
```

### Problema: Discord não recebe notificações

**Soluções:**

1. Verificar webhook configurado:
```bash
cat .env | grep DISCORD_WEBHOOK_URL
```

2. Testar webhook:
```bash
curl -X POST http://localhost:9094/test
```

3. Configurar webhook (se não configurado):
```bash
echo "DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/SEU_WEBHOOK" > .env
docker compose restart
```

### Problema: MLflow não mostra experimentos

**Soluções:**

1. Verificar se o PostgreSQL está rodando:
```bash
docker compose ps postgres
```

2. Verificar logs do MLflow:
```bash
docker compose logs mlflow
```

3. Reiniciar MLflow:
```bash
docker compose restart mlflow
```

### Problema: n8n não executa workflows

**Soluções:**

1. Verificar se os workflows estão ativos:
   - Abra http://localhost:5678
   - Verifique toggle "Active" em cada workflow

2. Verificar credenciais:
   - Workflows usam HTTP simples (sem autenticação)
   - Verificar URLs: `http://api:8000` (dentro do Docker)

3. Testar manualmente:
   - Abra o workflow
   - Clique em "Execute Workflow"
   - Veja erros no painel de execução

### Logs Úteis

```bash
# Ver logs de todos os serviços
docker compose logs -f

# Ver logs de um serviço específico
docker compose logs -f api
docker compose logs -f mlflow
docker compose logs -f n8n

# Ver últimas 100 linhas
docker compose logs --tail=100 api
```

---

## 🎓 Uso Acadêmico

Este projeto foi desenvolvido como parte de uma pós-graduação em Machine Learning e pode ser usado para:

### Apresentações
- Demonstrar pipeline MLOps completo
- Mostrar experimentação sistemática
- Explicar decisões de arquitetura

### Aprendizado
- Estudar código limpo e modular
- Entender boas práticas de MLOps
- Explorar notebooks com análises detalhadas

### Extensões Possíveis
- Adicionar mais features (indicadores técnicos)
- Testar outras arquiteturas (GRU, Transformer)
- Implementar ensemble de modelos
- Adicionar backtesting
- Criar dashboard de visualização

---

## 📚 Referências

- **FastAPI:** https://fastapi.tiangolo.com/
- **MLflow:** https://mlflow.org/
- **n8n:** https://n8n.io/
- **TensorFlow:** https://www.tensorflow.org/
- **Docker:** https://www.docker.com/
- **LSTM Paper:** Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.

---

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos e educacionais.

---

## 👤 Autor

Projeto desenvolvido como parte de Pós-Graduação em Machine Learning.

---

## 🚀 Quick Start

```bash
# 1. Extrair projeto
unzip stock-prediction.zip
cd stock-prediction

# 2. Iniciar serviços
docker compose up -d

# 3. Aguardar inicialização
sleep 30

# 4. Testar API
curl http://localhost:8000/health

# 5. Fazer predição
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"values": [100,102,101,103,105,104,106,108,107,109,111,110,112,114,113,115,117,116,118,120,119,121,123,122,124,126,125,127,129,128,130,132,131,133,135,134,136,138,137,139,141,140,142,144,143,145,147,146,148,150,149,151,153,152,154,156,155,157,159,158], "horizon": 5}'

# 6. Acessar interfaces
# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
# n8n: http://localhost:5678
```

**Pronto! O sistema está funcionando.** 🎉
