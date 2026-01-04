#!/bin/bash
# Script para validar a saúde da API e enviar alertas em caso de falha

set -e

API_URL="${API_URL:-http://localhost:8000}"
ALERTMANAGER_URL="${ALERTMANAGER_URL:-http://localhost:9093}"
MAX_RETRIES=3
RETRY_DELAY=5

echo "🔍 Validando saúde da API em $API_URL..."

# Função para enviar alerta
send_alert() {
    local severity=$1
    local summary=$2
    local description=$3
    
    echo "📢 Enviando alerta: $summary"
    
    curl -s -X POST "$ALERTMANAGER_URL/api/v2/alerts" \
        -H "Content-Type: application/json" \
        -d "[{
            \"labels\": {
                \"alertname\": \"APIValidationFailed\",
                \"severity\": \"$severity\",
                \"service\": \"stock-forecast-api\"
            },
            \"annotations\": {
                \"summary\": \"$summary\",
                \"description\": \"$description\"
            },
            \"startsAt\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
        }]" || echo "⚠️  Falha ao enviar alerta"
}

# Tentar conectar à API com retries
for i in $(seq 1 $MAX_RETRIES); do
    echo "Tentativa $i de $MAX_RETRIES..."
    
    if response=$(curl -s -f -m 10 "$API_URL/healthz" 2>&1); then
        status=$(echo "$response" | jq -r '.status' 2>/dev/null || echo "unknown")
        
        if [ "$status" = "healthy" ]; then
            echo "✅ API está saudável!"
            echo "📊 Resposta: $response"
            exit 0
        else
            echo "⚠️  API respondeu mas status não é 'healthy': $status"
        fi
    else
        echo "❌ Falha ao conectar à API"
    fi
    
    if [ $i -lt $MAX_RETRIES ]; then
        echo "⏳ Aguardando ${RETRY_DELAY}s antes de tentar novamente..."
        sleep $RETRY_DELAY
    fi
done

# Se chegou aqui, todas as tentativas falharam
echo "🚨 API não está respondendo após $MAX_RETRIES tentativas!"
send_alert "critical" \
    "API de predição não está respondendo" \
    "A API falhou em responder ao health check após $MAX_RETRIES tentativas"

exit 1
