FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY dockerfiles/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY api/ ./api/
COPY src/ ./src/

# Criar diretórios necessários
RUN mkdir -p /app/notebooks/artifacts /app/data /app/processed_data

# Expor porta
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
