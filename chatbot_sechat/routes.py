from flask import Blueprint, request, jsonify, session
import requests
import uuid
import os

from . import chatbot_sechat_bp

# URLs del endpoint Modal — actualizar después de hacer `modal deploy modal_app.py`
MODAL_CHAT_URL   = os.getenv('SECHAT_CHAT_URL', '')
MODAL_HEALTH_URL = os.getenv('SECHAT_HEALTH_URL', '')

# Historial de conversación por sesión (en memoria, se pierde al reiniciar)
conversations = {}


def get_or_create_session_id():
    if 'sechat_session_id' not in session:
        session['sechat_session_id'] = str(uuid.uuid4())
    return session['sechat_session_id']


@chatbot_sechat_bp.route('/health', methods=['GET'])
def health():
    """Comprueba si el endpoint Modal está activo."""
    if not MODAL_HEALTH_URL:
        return jsonify({'available': False, 'reason': 'URL no configurada'})
    try:
        resp = requests.get(MODAL_HEALTH_URL, timeout=10)
        return jsonify({'available': resp.status_code == 200})
    except Exception as e:
        return jsonify({'available': False, 'reason': str(e)})


@chatbot_sechat_bp.route('/chat', methods=['POST'])
def chat():
    """Recibe un mensaje del usuario, llama al endpoint Modal y devuelve la respuesta."""
    if not MODAL_CHAT_URL:
        return jsonify({'error': 'Servicio no configurado todavía.'}), 503

    data = request.get_json()
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({'error': 'Mensaje vacío'}), 400

    session_id = get_or_create_session_id()
    history = conversations.get(session_id, [])

    try:
        resp = requests.post(
            MODAL_CHAT_URL,
            json={'message': message, 'history': history},
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.Timeout:
        return jsonify({'error': 'El modelo tardó demasiado. Inténtalo de nuevo (puede estar en cold start).'}), 504
    except Exception as e:
        return jsonify({'error': f'Error conectando con el modelo: {str(e)}'}), 502

    if 'error' in result:
        return jsonify({'error': result['error']}), 500

    response_text = result.get('response', '')

    # Actualizar historial (máx 8 mensajes = 4 intercambios)
    history.append({'role': 'user',      'content': message})
    history.append({'role': 'assistant', 'content': response_text})
    conversations[session_id] = history[-4:]

    return jsonify({'response': response_text})


@chatbot_sechat_bp.route('/clear', methods=['POST'])
def clear():
    """Limpia el historial de conversación de la sesión."""
    session_id = get_or_create_session_id()
    conversations.pop(session_id, None)
    session.pop('sechat_session_id', None)
    return jsonify({'ok': True})
