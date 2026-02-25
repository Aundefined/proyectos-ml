from flask import render_template, request, jsonify, session
import uuid
import requests
import os

from . import chatbot_gemelo_bp

GEMELO_API_URL = os.getenv('GEMELO_API_URL', 'https://8dvxk6b5n2.execute-api.eu-west-1.amazonaws.com')


def get_or_create_session_id():
    """Obtiene o crea un session_id único por usuario, que identifica la conversación en AWS."""
    if 'chat_gemelo_session_id' not in session:
        session['chat_gemelo_session_id'] = str(uuid.uuid4())
    return session['chat_gemelo_session_id']


@chatbot_gemelo_bp.route('/')
def index_route():
    """Página principal del chatbot Gemelo Digital."""
    return render_template('chatbot_gemelo.html')


@chatbot_gemelo_bp.route('/chat', methods=['POST'])
def chat():
    """Envía el mensaje a la API del Gemelo Digital en AWS y devuelve la respuesta."""
    try:
        data = request.get_json()
        question = data.get('message', '').strip()

        if not question:
            return jsonify({'error': 'Mensaje vacío'}), 400

        session_id = get_or_create_session_id()

        response = requests.post(
            f'{GEMELO_API_URL}/chat',
            json={'message': question, 'session_id': session_id},
            timeout=30
        )
        response.raise_for_status()

        result = response.json()

        # Sincronizar el session_id por si la API devuelve uno distinto
        if 'session_id' in result:
            session['chat_gemelo_session_id'] = result['session_id']

        return jsonify({'response': result.get('response', 'Sin respuesta del servidor.')})

    except requests.exceptions.Timeout:
        print('❌ Timeout al contactar la API del Gemelo Digital')
        return jsonify({'error': 'El servidor tardó demasiado en responder. Inténtalo de nuevo.'}), 504
    except requests.exceptions.ConnectionError:
        print('❌ Error de conexión con la API del Gemelo Digital')
        return jsonify({'error': 'No se pudo conectar con el servidor. Inténtalo de nuevo.'}), 503
    except requests.exceptions.HTTPError as e:
        print(f'❌ HTTP error de la API del Gemelo Digital: {e}')
        return jsonify({'error': 'Error en el servidor remoto. Inténtalo más tarde.'}), 502
    except Exception as e:
        print(f'❌ Error inesperado en /chat gemelo: {e}')
        return jsonify({'error': 'Ha ocurrido un error interno. Por favor, inténtalo de nuevo.'}), 500


@chatbot_gemelo_bp.route('/clear-chat', methods=['POST'])
def clear_chat():
    """Limpia la sesión: el próximo mensaje abrirá una conversación nueva en AWS."""
    try:
        session.pop('chat_gemelo_session_id', None)
        return jsonify({'success': True})
    except Exception as e:
        print(f'❌ Error limpiando chat gemelo: {e}')
        return jsonify({'error': 'Error al limpiar el chat'}), 500
