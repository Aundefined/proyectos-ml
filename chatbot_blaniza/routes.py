from flask import Blueprint, render_template, request, jsonify, session
import torch
import uuid
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from . import chatbot_blaniza_bp

warnings.filterwarnings("ignore")

# Variables globales para el modelo
model = None
tokenizer = None

# Diccionario para almacenar conversaciones por sesión
conversations = {}

def load_blaniza_model():
    """Cargar el modelo Blaniza Assistant desde Hugging Face"""
    global model, tokenizer
    
    try:
        model_name = "ArnaudClaudeML/blaniza-assistant"
        print("=" * 60)
        print("🚀 INICIANDO CARGA DEL MODELO BLANIZA ASSISTANT")
        print("=" * 60)
        print(f"📦 Modelo: {model_name}")
        
        # Configurar dispositivo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Dispositivo detectado: {device}")
        
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"💾 GPU: {gpu_name}")
            print(f"💾 VRAM total: {total_memory:.1f} GB")
            
            print("⚙️ Configurando cuantización 4-bit...")
            # Configuración de cuantización para ahorrar memoria
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            print("✅ Configuración de cuantización lista")
            
            print("📥 Descargando y cargando modelo con cuantización 4-bit...")
            print("   (Esto puede tomar varios minutos en la primera carga)")
            # Cargar modelo con cuantización
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                revision="main"
            )
            
            # Mostrar uso de memoria después de cargar
            used_memory = torch.cuda.memory_allocated() / (1024**3)
            print(f"💾 VRAM utilizada: {used_memory:.2f} GB")
            print(f"💾 VRAM libre: {total_memory - used_memory:.1f} GB")
            
        else:
            print("💻 Cargando en CPU (sin cuantización)...")
            # Cargar en CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
        
        print("📝 Cargando tokenizer...")
        # Cargar tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ Tokenizer cargado. Vocabulario: {len(tokenizer)} tokens")
        
        # Información del modelo
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Parámetros del modelo: {total_params:,}")
        
        print("=" * 60)
        print("🎉 MODELO BLANIZA ASSISTANT CARGADO EXITOSAMENTE")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print("❌ ERROR AL CARGAR EL MODELO BLANIZA")
        print("=" * 60)
        print(f"Error: {e}")
        print("Posibles causas:")
        print("- Conexión a internet")
        print("- Espacio insuficiente en disco")
        print("- Problema con las dependencias (transformers, torch, bitsandbytes)")
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

def generate_answer_with_blaniza(messages, max_tokens=150):
    """Generar respuesta usando el modelo Blaniza"""
    try:
        print("🤖 Iniciando generación de respuesta con modelo Blaniza")
        
        # Construir el prompt en formato conversacional
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            prompt += f"{role}: {content}\n"
        
        prompt += "assistant:"
        
        print(f"📝 Longitud del prompt: {len(prompt)} caracteres")
        
        # Tokenizar
        print("🔤 Tokenizando prompt...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_tokens = inputs.input_ids.shape[1]
        print(f"📊 Tokens de entrada: {input_tokens}")
        
        # Generar respuesta
        print(f"⚡ Generando respuesta (máx {max_tokens} tokens)...")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        # Extraer solo la respuesta generada
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        output_tokens = outputs[0].shape[0] - input_tokens
        print(f"📊 Tokens generados: {output_tokens}")
        print(f"✅ Respuesta generada exitosamente: {len(response)} caracteres")
        
        return response
        
    except Exception as e:
        print(f"❌ Error generando respuesta con Blaniza: {e}")
        return "Lo siento, ha ocurrido un error al procesar tu pregunta. Por favor, inténtalo de nuevo."

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
    
    # Cargar modelo si no está cargado
    if model is None or tokenizer is None:
        print("⚠️ Modelo no cargado, iniciando carga...")
        if not load_blaniza_model():
            print("❌ Fallo al cargar el modelo, mostrando página de error")
            return render_template('chatbot_blaniza.html', error="Error: No se pudo cargar el modelo Blaniza Assistant")
    else:
        print("✅ Modelo ya está cargado y listo")
    
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
        
        # Asegurar que el modelo está cargado
        if model is None or tokenizer is None:
            print("⚠️ Modelo no disponible, intentando cargar...")
            if not load_blaniza_model():
                print("❌ Error: no se pudo cargar el modelo")
                return jsonify({'error': 'Error del sistema. El modelo Blaniza no está disponible.'}), 500
        
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