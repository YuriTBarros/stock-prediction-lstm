#!/bin/bash

echo "=========================================="
echo "Aplicando Correções Críticas MLOps"
echo "=========================================="
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Função para imprimir com cor
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Verificar se docker compose está disponível
if ! command -v docker &> /dev/null; then
    print_error "Docker não está instalado!"
    exit 1
fi

print_status "Docker encontrado"

# Passo 1: Parar containers antigos
echo ""
echo "Passo 1: Parando containers antigos..."
docker compose down -v
if [ $? -eq 0 ]; then
    print_status "Containers parados e volumes removidos"
else
    print_error "Erro ao parar containers"
    exit 1
fi

# Passo 2: Reconstruir imagens
echo ""
echo "Passo 2: Reconstruindo imagens..."
print_warning "Isso pode levar alguns minutos..."
docker compose build --no-cache
if [ $? -eq 0 ]; then
    print_status "Imagens reconstruídas"
else
    print_error "Erro ao reconstruir imagens"
    exit 1
fi

# Passo 3: Iniciar containers
echo ""
echo "Passo 3: Iniciando containers..."
docker compose up -d
if [ $? -eq 0 ]; then
    print_status "Containers iniciados"
else
    print_error "Erro ao iniciar containers"
    exit 1
fi

# Aguardar serviços ficarem prontos
echo ""
echo "Aguardando serviços ficarem prontos..."
sleep 10

# Passo 4: Verificar serviços
echo ""
echo "Passo 4: Verificando serviços..."

# PostgreSQL
echo -n "PostgreSQL... "
if docker compose exec -T postgres pg_isready -U mlops &> /dev/null; then
    print_status "OK"
else
    print_error "FALHOU"
fi

# MLflow
echo -n "MLflow... "
if curl -s http://localhost:5000 > /dev/null 2>&1; then
    print_status "OK"
else
    print_error "FALHOU (pode levar mais tempo para iniciar)"
fi

# API
echo -n "API... "
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "OK"
else
    print_error "FALHOU (pode levar mais tempo para iniciar)"
fi

# n8n
echo -n "n8n... "
if curl -s http://localhost:5678 > /dev/null 2>&1; then
    print_status "OK"
else
    print_error "FALHOU"
fi

# Alertmanager
echo -n "Alertmanager... "
if curl -s http://localhost:9093 > /dev/null 2>&1; then
    print_status "OK"
else
    print_error "FALHOU"
fi

# Discord Proxy
echo -n "Discord Proxy... "
if curl -s http://localhost:9094/health > /dev/null 2>&1; then
    print_status "OK"
else
    print_error "FALHOU"
fi

# Resumo
echo ""
echo "=========================================="
echo "Correções Aplicadas!"
echo "=========================================="
echo ""
echo "Acesse os serviços:"
echo "  • MLflow UI:      http://localhost:5000"
echo "  • n8n:            http://localhost:5678"
echo "  • API Docs:       http://localhost:8000/docs"
echo "  • Alertmanager:   http://localhost:9093"
echo ""
echo "Próximos passos:"
echo "  1. Acesse n8n e crie uma conta (primeira vez)"
echo "  2. Teste a API: curl http://localhost:8000/health"
echo "  3. Execute um experimento (veja CRITICAL_FIXES.md)"
echo ""
echo "Ver logs:"
echo "  docker compose logs -f"
echo ""
