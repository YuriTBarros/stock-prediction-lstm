# Tech Challenge 4: Previsão de Preços de Ações com LSTM

## Pós-Graduação em Machine Learning Engineering

**Integrantes do Grupo:**
* [Nome do Integrante 1]
* [Nome do Integrante 2]
* [Nome do Integrante 3]

---

### 1. Descrição do Problema

Este projeto, parte do Tech Challenge da Fase 04, visa criar um pipeline completo de Machine Learning para prever o valor de fechamento de ações. Utilizamos uma rede neural **Long Short-Term Memory (LSTM)** construída com **PyTorch**. Todo o ciclo de experimentação é gerenciado com **MLflow**.

O projeto inclui três componentes principais:
1.  **Pipeline de Treinamento:** Scripts para coleta de dados, treinamento e versionamento do modelo.
2.  **Dashboard de Prototipagem:** Uma aplicação interativa com **Streamlit** para visualização e teste das previsões.
3.  **API de Deploy:** Uma API RESTful conteinerizada com **Docker** para servir o modelo em produção.

### 2. Stack de Tecnologias

* **Coleta de Dados:** `yfinance`
* **Modelagem e Treinamento:** `Python`, `PyTorch`, `Scikit-learn`
* **Experimentação e MLOps:** `MLflow`
* **Prototipagem e Visualização:** `Streamlit`
* **Deploy:** `FastAPI` (ou `Flask`), `Docker`

### 3. Estrutura do Projeto
```
stock-prediction-lstm/
│
├── api/
│   ├── main.py             # Lógica da API (FastAPI/Flask)
│   └── Dockerfile          # Container Docker para a API
│
├── dashboard/
│   └── app.py              # Código do dashboard interativo com Streamlit
│
├── notebooks/
│   └── data_exploration_and_modeling.ipynb # Notebook para análise e experimentação
│
├── src/
│   ├── data_ingestion.py   # Script para baixar os dados
│   ├── preprocessing.py    # Script para pré-processar os dados
│   ├── model.py            # Definição do modelo LSTM em PyTorch (nn.Module)
│   ├── train.py            # Script para treinar o modelo e logar com MLflow
│   └── predict.py          # Script que carrega o modelo do MLflow para fazer previsões
│
├── .gitignore              # Arquivos a serem ignorados (incluindo /mlruns)
├── README.md               # Documentação do projeto
└── requirements.txt        # Dependências (incluindo streamlit)
```

### 4. Como Executar o Projeto

1.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Execute o treinamento:**
    Isso irá buscar os dados, treinar o modelo e registrar tudo no MLflow.
    ```bash
    python src/train.py
    ```

3.  **Visualize os experimentos no MLflow:**
    ```bash
    mlflow ui
    ```
    Acesse `http://localhost:5000` no seu navegador.

4.  **Execute o Dashboard Interativo:**
    Para visualizar e interagir com as previsões do modelo.
    ```bash
    streamlit run dashboard/app.py
    ```

5.  **Construa e execute a API com Docker:**
    ```bash
    cd api/
    docker build -t stock-prediction-api .
    docker run -p 8000:8000 stock-prediction-api
    ```

### 5. Dashboard de Prototipagem com Streamlit

Para facilitar a demonstração e a validação do modelo, foi desenvolvido um dashboard interativo. Esta aplicação permite que o usuário:
* Visualize os dados históricos utilizados no treinamento.
* Veja a performance do modelo comparando valores previstos vs. reais.
* Realize novas previsões de forma interativa.

### 6. Entregáveis

#### 6.1. API em Produção
* **Link da API:** [Inserir o link para a API em produção, caso tenha sido deployada em um ambiente de nuvem]

#### 6.2. Vídeo de Apresentação
* **Link do Vídeo:** [Inserir o link para o vídeo de apresentação da API e do Dashboard]

### 7. API de Deploy

A API RESTful, desenvolvida com **[Flask ou FastAPI]**, carrega o modelo versionado pelo MLflow para servir as previsões.

#### Endpoint: `POST /predict`

* **Corpo da Requisição (JSON):**
    ```json
    {
      "historical_data": [150.5, 151.2, 150.8, ...]
    }
    ```
* **Resposta (JSON):**
    ```json
    {
      "prediction": 152.3
    }
    ```

### 8. Monitoramento

[Descrever as ferramentas e estratégias de monitoramento configuradas para rastrear a performance do modelo em produção, como tempo de resposta e utilização de recursos, conforme solicitado].
