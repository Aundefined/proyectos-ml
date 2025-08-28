from flask import Blueprint, render_template, request, jsonify, session
import uuid
import requests
import json

from . import chatbot_blaniza_bp

# URL del Space de Hugging Face (usaremos la API de Gradio)
BLANIZA_SPACE_URL = "https://arnaudclaudeml-blaniza-assistant.hf.space"

# Diccionario para almacenar conversaciones por sesión
conversations = {}

def check_blaniza_service():
    """Verificar si el servicio Blaniza está disponible"""
    try:
        # Verificar que el Space responda
        print(f"🔍 Verificando Space en: {BLANIZA_SPACE_URL}")
        
        response = requests.get(BLANIZA_SPACE_URL, timeout=10)
        print(f"📡 Respuesta del Space: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Space está activo")
            return True
        else:
            print(f"⚠️ Space no disponible: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout conectando al Space ({BLANIZA_SPACE_URL})")
        return False
    except requests.exceptions.ConnectionError:
        print(f"🌐 Error de conexión al Space ({BLANIZA_SPACE_URL})")
        return False
    except Exception as e:
        print(f"❌ Error inesperado verificando Space: {type(e).__name__}: {e}")
        return False

def get_system_prompt():
    """Obtener el system prompt para el modelo Blaniza"""
    return """Eres un asistente útil, educado y conciso. Respondes siempre en español.

Tu misión exclusiva es ayudar con el manual del Sistema de gestión de pedidos de Logística Blaniza, que incluye únicamente los siguientes temas:
- Instalación de Node.js y NPM
- Configuración y uso de la aplicación
- Variables de entorno y configuración
- API y funcionalidades del sistema
- Solución de errores y troubleshooting
- Sistema de gestión web (PHP/Laravel)
- Base de datos MongoDB
- Mantenimiento de usuarios y perfiles

Restricciones:
1. No debes responder, improvisar ni especular sobre temas fuera de este manual (ej: cocina, deportes, política, historia, filosofía, temas personales, chistes, etc.).
2. Si una pregunta está fuera de este ámbito, responde educadamente: "Lo siento, solo puedo ayudarte con temas relacionados con el manual del Sistema de gestión de pedidos de Logística Blaniza."
3. Si la pregunta es ambigua o poco clara, pide amablemente que la reformulen para poder ayudar mejor.
4. Si no encuentras la respuesta en el manual, responde claramente que no tienes información suficiente en el documento.

Tu rol es actuar siempre como experto en este manual, nada más."""

def generate_answer_with_blaniza(messages, max_tokens=1000):
    """Generar respuesta usando la API de Gradio del Space Blaniza"""
    try:
        print(f"🤖 Enviando petición al Space: {BLANIZA_SPACE_URL}")
        print(f"📦 Payload: {len(messages)} mensajes, max_tokens={max_tokens}")
        
        # Gradio Client para conectar con el Space
        from gradio_client import Client
        
        import time
        start_time = time.time()
        
        # Conectar con el Space (con timeout reducido)
        client = Client(BLANIZA_SPACE_URL)
        
        # Mostrar información de debug del Space (solo si hay problemas)
        # print("📋 Información del Space:")
        # try:
        #     api_info = client.view_api()
        #     print(api_info)
        # except Exception as debug_e:
        #     print(f"No se pudo obtener info del API: {debug_e}")
        
        # Solo enviar el último mensaje del usuario para la interfaz simple
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if not user_message:
            return "No se encontró mensaje del usuario."
        
        print(f"📩 Enviando mensaje: {user_message[:50]}...")
        
        try:
            result = client.predict(user_message, api_name="/simple_chat")
            
        except Exception as e1:
            print(f"⚠️ Error con /simple_chat: {e1}")
            return f"Error del modelo en el Space: {str(e1)}"
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo de respuesta: {elapsed_time:.2f}s")
        print(f"✅ Respuesta recibida: {len(str(result))} caracteres")
        
        return str(result)
            
    except ImportError:
        print("❌ Error: gradio_client no está instalado")
        return "Error: Dependencia faltante. Instala gradio_client."
    except Exception as e:
        print(f"❌ Error conectando con Space: {type(e).__name__}: {e}")
        return f"Lo siento, no se pudo conectar con el servicio. Error: {str(e)}"

def get_or_create_session_id():
    """Obtiene o crea un ID de sesión único"""
    if 'chat_blaniza_session_id' not in session:
        session['chat_blaniza_session_id'] = str(uuid.uuid4())
    return session['chat_blaniza_session_id']

def get_conversation_history(session_id):
    """Obtiene el historial de conversación para una sesión"""
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]

@chatbot_blaniza_bp.route('/')
def index_route():
    """Página principal del chatbot Blaniza"""
    print("🌐 Acceso a la página principal del Chatbot Blaniza")
    
    # Verificar si el servicio está disponible
    service_status = check_blaniza_service()
    print(f"ℹ️ Estado del servicio Blaniza: {'✅ Disponible' if service_status else '❌ No disponible'}")
    
    return render_template('chatbot_blaniza.html')

@chatbot_blaniza_bp.route('/chat', methods=['POST'])
def chat():
    """Endpoint para procesar mensajes del chat con servicio Blaniza"""
    try:
        print("💬 Nueva petición de chat recibida")
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            print("⚠️ Mensaje vacío recibido")
            return jsonify({'error': 'Mensaje vacío'}), 400
        
        print(f"📩 Pregunta del usuario: {question[:50]}{'...' if len(question) > 50 else ''}")
        
        # Verificar que el servicio esté disponible
        print("🔍 Verificando disponibilidad del servicio...")
        if not check_blaniza_service():
            print("❌ Error: el servicio Blaniza no está disponible")
            return jsonify({'error': 'El servicio Blaniza no está disponible en este momento. Por favor, inténtalo más tarde.'}), 503
        print("✅ Servicio verificado y disponible")
        
        # Obtener o crear sesión
        session_id = get_or_create_session_id()
        messages = get_conversation_history(session_id)
        
        # Si es el primer mensaje de la sesión, añadir system prompt
        if not messages:
            system_prompt = get_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
        
        # Añadir el mensaje del usuario al historial
        messages.append({"role": "user", "content": question})
        
        # Generar respuesta usando el servicio API
        answer = generate_answer_with_blaniza(messages)
        
        # Añadir la respuesta del asistente al historial
        messages.append({"role": "assistant", "content": answer})
        
        # Actualizar el historial en la sesión
        conversations[session_id] = messages
        
        # Preparar información de debug para el frontend
        debug_info = {
            'session_id': session_id[:8],  # Solo los primeros 8 caracteres por privacidad
            'messages_count': len(messages),
            'messages_sent': len(messages) - 1,  # Sin contar la respuesta recién añadida
            'conversation_history': [
                {
                    'role': msg['role'], 
                    'content': msg['content'][:100] + '...' if len(msg['content']) > 100 else msg['content']
                } 
                for msg in messages[:-1]  # Sin incluir la respuesta recién añadida
            ],
            'last_user_message': question,
            'response_length': len(answer)
        }
        
        return jsonify({
            'response': answer,
            'debug_info': debug_info
        })
        
    except Exception as e:
        print(f"Error en /chat: {e}")
        return jsonify({'error': 'Ha ocurrido un error interno. Por favor, inténtalo de nuevo.'}), 500

@chatbot_blaniza_bp.route('/clear-chat', methods=['POST'])
def clear_chat():
    """Endpoint para limpiar el historial de chat"""
    try:
        session_id = get_or_create_session_id()
        if session_id in conversations:
            del conversations[session_id]
        # También limpiar la sesión
        session.pop('chat_blaniza_session_id', None)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error limpiando chat: {e}")
        return jsonify({'error': 'Error al limpiar el chat'}), 500