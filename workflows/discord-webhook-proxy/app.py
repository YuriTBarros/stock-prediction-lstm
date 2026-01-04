"""
Proxy para converter alertas do Alertmanager em mensagens Discord
Permite configurar webhook do Discord via variável de ambiente
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

import requests
from flask import Flask, request, jsonify

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuração do webhook Discord (via variável de ambiente)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Cores para diferentes severidades
SEVERITY_COLORS = {
    "critical": 0xFF0000,  # Vermelho
    "warning": 0xFFA500,   # Laranja
    "info": 0x00FF00,      # Verde
    "success": 0x00FF00    # Verde
}

# Emojis para diferentes componentes
COMPONENT_EMOJIS = {
    "experiment": "🧪",
    "training": "🎓",
    "testing": "✅",
    "model": "🤖",
    "api": "🔌",
    "database": "💾"
}


def format_alert_for_discord(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Formatar alerta do Alertmanager para mensagem Discord"""
    
    # Extrair informações do alerta
    labels = alert.get("labels", {})
    annotations = alert.get("annotations", {})
    status = alert.get("status", "firing")
    
    alertname = labels.get("alertname", "Unknown Alert")
    severity = labels.get("severity", "info")
    component = labels.get("component", "system")
    
    summary = annotations.get("summary", alertname)
    description = annotations.get("description", "Sem descrição")
    
    # Emoji baseado no componente
    emoji = COMPONENT_EMOJIS.get(component, "📊")
    
    # Cor baseada na severidade
    color = SEVERITY_COLORS.get(severity, 0x808080)
    
    # Status emoji
    status_emoji = "🔴" if status == "firing" else "✅"
    
    # Construir embed do Discord
    embed = {
        "title": f"{emoji} {status_emoji} {alertname}",
        "description": description,
        "color": color,
        "fields": [
            {
                "name": "Severidade",
                "value": severity.upper(),
                "inline": True
            },
            {
                "name": "Componente",
                "value": component.capitalize(),
                "inline": True
            },
            {
                "name": "Status",
                "value": status.upper(),
                "inline": True
            }
        ],
        "footer": {
            "text": "MLOps Stock Prediction"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Adicionar campos extras dos labels
    extra_fields = []
    for key, value in labels.items():
        if key not in ["alertname", "severity", "component"]:
            extra_fields.append({
                "name": key.replace("_", " ").title(),
                "value": str(value),
                "inline": True
            })
    
    if extra_fields:
        embed["fields"].extend(extra_fields[:5])  # Máximo 5 campos extras
    
    return embed


def send_to_discord(embeds: List[Dict[str, Any]]) -> bool:
    """Enviar mensagem para Discord"""
    
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL não configurado. Alerta não será enviado ao Discord.")
        return False
    
    try:
        payload = {
            "username": "MLOps Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
            "embeds": embeds
        }
        
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code in [200, 204]:
            logger.info(f"Mensagem enviada ao Discord com sucesso")
            return True
        else:
            logger.error(f"Erro ao enviar para Discord: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Erro ao enviar para Discord: {e}")
        return False


@app.route('/webhook', methods=['POST'])
def discord_webhook():
    """Endpoint para receber alertas do Alertmanager e enviar para Discord"""
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        logger.info(f"Alerta recebido: {json.dumps(data, indent=2)}")
        
        # Processar alertas
        alerts = data.get("alerts", [])
        
        if not alerts:
            return jsonify({"message": "No alerts to process"}), 200
        
        # Converter alertas para embeds do Discord
        embeds = []
        for alert in alerts[:10]:  # Discord permite máximo 10 embeds por mensagem
            embed = format_alert_for_discord(alert)
            embeds.append(embed)
        
        # Enviar para Discord
        if DISCORD_WEBHOOK_URL:
            success = send_to_discord(embeds)
            if success:
                return jsonify({"message": f"{len(embeds)} alertas enviados ao Discord"}), 200
            else:
                return jsonify({"error": "Failed to send to Discord"}), 500
        else:
            logger.warning("Discord webhook não configurado. Alertas apenas logados.")
            return jsonify({"message": "Discord webhook not configured, alerts logged only"}), 200
    
    except Exception as e:
        logger.error(f"Erro ao processar webhook: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "service": "discord-webhook-proxy",
        "discord_configured": bool(DISCORD_WEBHOOK_URL),
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/test', methods=['POST'])
def test_notification():
    """Testar notificação Discord"""
    
    if not DISCORD_WEBHOOK_URL:
        return jsonify({
            "error": "DISCORD_WEBHOOK_URL não configurado",
            "help": "Configure a variável de ambiente DISCORD_WEBHOOK_URL no docker-compose.yml"
        }), 400
    
    # Criar alerta de teste
    test_alert = {
        "status": "firing",
        "labels": {
            "alertname": "TestNotification",
            "severity": "info",
            "component": "system"
        },
        "annotations": {
            "summary": "Teste de Notificação Discord",
            "description": "Esta é uma mensagem de teste para verificar a integração com o Discord."
        }
    }
    
    embed = format_alert_for_discord(test_alert)
    success = send_to_discord([embed])
    
    if success:
        return jsonify({"message": "Notificação de teste enviada com sucesso!"}), 200
    else:
        return jsonify({"error": "Falha ao enviar notificação de teste"}), 500


if __name__ == '__main__':
    if DISCORD_WEBHOOK_URL:
        logger.info("Discord webhook configurado ✅")
    else:
        logger.warning("⚠️  DISCORD_WEBHOOK_URL não configurado!")
        logger.warning("Configure a variável de ambiente para receber notificações no Discord")
    
    app.run(host='0.0.0.0', port=9094, debug=False)
