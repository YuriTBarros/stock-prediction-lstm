-- Inicialização dos bancos de dados para o projeto MLOps

-- Criar banco de dados para MLflow
CREATE DATABASE mlflow;

-- Conceder permissões ao usuário mlops
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlops;
