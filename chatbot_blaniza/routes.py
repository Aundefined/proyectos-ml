from flask import Blueprint, render_template, request, jsonify, session
import torch
import uuid
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import chatbot_blaniza_bp

warnings.filterwarnings("ignore")

# Variables globales para el modelo
model = None
tokenizer = None

# Diccionario para almacenar conversaciones por sesión
conversations = {}

def load_blaniza_model():
    """Cargar el modelo Blaniza Assistant optimizado para CPU y memoria limitada"""
    global model, tokenizer
    
    try:
        model_name = "ArnaudClaudeML/blaniza-assistant"
        print("=" * 60)
        print("INICIANDO CARGA OPTIMIZADA DEL MODELO BLANIZA")
        print("=" * 60)
        print(f"📦 Modelo: {model_name}")
        
        # Configurar dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Dispositivo detectado: {device}")
        
        # Cargar siempre en CPU para Railway (más rápido y estable)
        print("💻 Cargando en CPU (optimizado para Railway)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,  # Usar float32 para estabilidad
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        
        print("⚙️ Aplicando optimizaciones...")
        model.eval()
        
        print("📝 Cargando tokenizer...")
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer cargado. Vocabulario: {len(tokenizer)} tokens")
        
        # Información del modelo
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Parámetros del modelo: {total_params:,}")
        
        # Mostrar información de memoria (aproximada)
        import psutil
        memory_info = psutil.virtual_memory()
        print(f"💾 RAM total del sistema: {memory_info.total / (1024**3):.1f} GB")
        print(f"💾 RAM disponible: {memory_info.available / (1024**3):.1f} GB")
        
        print("=" * 60)
        print("🎉 MODELO BLANIZA CARGADO CON OPTIMIZACIONES")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print("❌ ERROR AL CARGAR EL MODELO BLANIZA")
        print("=" * 60)
        print(f"Error: {e}")
        print("Posibles causas:")
        print("- Memoria RAM insuficiente (Railway limita a 8GB)")
        print("- Conexión a internet")
        print("- Espacio insuficiente en disco")
        print("- Problema con las dependencias")
        print("=" * 60)
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

def generate_answer_with_blaniza(messages, max_tokens=200):  # Aumentar tokens para respuestas completas
    """Generar respuesta usando el modelo Blaniza optimizado para CPU"""
    try:
        print("🤖 Iniciando generación optimizada con modelo Blaniza")
        
        # Construir el prompt en formato conversacional (más eficiente)
        prompt = ""
        # Limitar el historial para ahorrar memoria
        recent_messages = messages[-5:] if len(messages) > 5 else messages
        
        for message in recent_messages:
            role = message["role"]
            content = message["content"]
            # Truncar mensajes muy largos
            if len(content) > 500:
                content = content[:500] + "..."
            prompt += f"{role}: {content}\n"
        
        prompt += "assistant:"
        
        print(f"📝 Longitud del prompt: {len(prompt)} caracteres")
        print(f"📝 Mensajes en el historial: {len(recent_messages)}")
        
        # Tokenizar con longitud máxima equilibrada
        print("🔤 Tokenizando prompt...")
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=768,  # Permitir más contexto para respuestas completas
            truncation=True
        )
        
        # No mover a GPU si no existe
        if model.device.type != 'cpu':
            inputs = inputs.to(model.device)
            
        input_tokens = inputs.input_ids.shape[1]
        print(f"📊 Tokens de entrada: {input_tokens}")
        
        # Verificar memoria disponible antes de generar
        import psutil
        memory_before = psutil.virtual_memory().percent
        print(f"💾 Uso de memoria antes: {memory_before:.1f}%")
        
        # Parámetros de generación equilibrados para respuestas completas
        generation_params = {
            "max_new_tokens": min(max_tokens, 150),  # Permitir respuestas más largas
            "do_sample": True,
            "top_p": 0.9,  # Restaurar diversidad para mejores respuestas
            "temperature": 0.7,  # Equilibrio entre creatividad y eficiencia
            "pad_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # Penalización normal
            "early_stopping": False,  # No cortar respuestas prematuramente
            "use_cache": True,  # Usar cache si está disponible
            "eos_token_id": tokenizer.eos_token_id,  # Asegurar final natural
        }
        
        print(f"⚡ Generando respuesta optimizada (máx {generation_params['max_new_tokens']} tokens)...")
        
        # Generar con manejo de memoria
        with torch.no_grad():  # Importante: no calcular gradientes
            try:
                outputs = model.generate(
                    inputs.input_ids,
                    **generation_params
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("⚠️ Memoria insuficiente, usando parámetros reducidos...")
                    generation_params["max_new_tokens"] = 80  # Aún permitir respuestas razonables
                    generation_params["temperature"] = 0.5
                    generation_params["early_stopping"] = True  # Solo activar en emergencia
                    outputs = model.generate(
                        inputs.input_ids,
                        **generation_params
                    )
                else:
                    raise e
        
        # Extraer solo la respuesta generada
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        output_tokens = outputs[0].shape[0] - input_tokens
        memory_after = psutil.virtual_memory().percent
        
        print(f"📊 Tokens generados: {output_tokens}")
        print(f"💾 Uso de memoria después: {memory_after:.1f}%")
        print(f"✅ Respuesta generada: {len(response)} caracteres")
        
        # Limpiar memoria si es necesario
        if memory_after > 85:
            import gc
            gc.collect()
        
        return response
        
    except Exception as e:
        print(f"❌ Error generando respuesta: {e}")
        print(f"❌ Tipo de error: {type(e).__name__}")
        
        # Limpiar memoria en caso de error
        import gc
        gc.collect()
            
        return "Lo siento, ha ocurrido un error de memoria al procesar tu pregunta. Intenta con una pregunta más corta."

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
    
    # NO cargar el modelo al inicio para ahorrar memoria
    # El modelo se cargará de forma lazy cuando se envíe el primer mensaje
    print("ℹ️ Modo lazy loading: El modelo se cargará cuando envíes tu primer mensaje")
    
    return render_template('chatbot_blaniza.html')

@chatbot_blaniza_bp.route('/chat', methods=['POST'])
def chat():
    """Endpoint para procesar mensajes del chat con modelo Blaniza"""
    try:
        print("💬 Nueva petición de chat recibida")
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            print("⚠️ Mensaje vacío recibido")
            return jsonify({'error': 'Mensaje vacío'}), 400
        
        print(f"📩 Pregunta del usuario: {question[:50]}{'...' if len(question) > 50 else ''}")
        
        # Cargar modelo de forma lazy (solo cuando se necesita)
        if model is None or tokenizer is None:
            print("⚠️ Cargando modelo por primera vez (lazy loading)...")
            if not load_blaniza_model():
                print("❌ Error: no se pudo cargar el modelo")
                return jsonify({'error': 'Error del sistema. No hay suficiente memoria para cargar el modelo Blaniza. Intenta reiniciar la aplicación.'}), 500
        
        # Obtener o crear sesión
        session_id = get_or_create_session_id()
        messages = get_conversation_history(session_id)
        
        # Si es el primer mensaje de la sesión, añadir system prompt
        if not messages:
            system_prompt = get_system_prompt()
            messages.append({"role": "system", "content": system_prompt})
        
        # Añadir el mensaje del usuario al historial
        messages.append({"role": "user", "content": question})
        
        # Generar respuesta con el modelo Blaniza
        answer = generate_answer_with_blaniza(messages)
        
        # Añadir la respuesta del asistente al historial
        messages.append({"role": "assistant", "content": answer})
        
        # Actualizar el historial en la sesión
        conversations[session_id] = messages
        
        return jsonify({'response': answer})
        
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