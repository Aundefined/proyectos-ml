from flask import Blueprint, render_template, request, jsonify, session
from openai import OpenAI
import numpy as np
import faiss
import pickle
import os
import uuid

from . import chatbot_pedidos_bp

# ✅ Configura OpenAI usando variable de entorno
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Variables globales para el índice y chunks
index = None
chunks = None

# Diccionario para almacenar conversaciones por sesión
conversations = {}

def load_chatbot_data():
    """Carga el índice FAISS y los chunks desde disco"""
    global index, chunks
    
    try:
        index_file = "attached-files/pedidos_index.index"
        chunks_file = "attached-files/pedidos_chunks.pkl"
        
        if os.path.exists(index_file) and os.path.exists(chunks_file):
            print("🔁 Cargando índice y fragmentos desde disco...")
            index = faiss.read_index(index_file)
            with open(chunks_file, "rb") as f:
                chunks = pickle.load(f)
            return True
        else:
            print("❌ No se encontraron los archivos de índice y chunks")
            return False
    except Exception as e:
        print(f"❌ Error al cargar datos del chatbot: {e}")
        return False

def get_embeddings(texts):
    """Genera embeddings usando OpenAI"""
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [np.array(e.embedding) for e in response.data]
    except Exception as e:
        print(f"Error generando embeddings: {e}")
        return None

def search_similar_chunks(question, k=3):
    """Busca chunks similares usando FAISS"""
    global index, chunks
    
    if index is None or chunks is None:
        return []
    
    try:
        question_embedding = get_embeddings([question])[0]
        distances, indices = index.search(np.array([question_embedding]), k)
        return [chunks[i] for i in indices[0]]
    except Exception as e:
        print(f"Error en búsqueda semántica: {e}")
        return []

def get_system_prompt(context):
    """Genera el prompt del sistema con el contexto"""
    return f"""Actúa como un experto asistente que responde preguntas sobre el "Manual de Usuario Aplicaciones del Sistema de gestión de pedidos de Logística Blaniza", usando única y exclusivamente el siguiente contexto:

---contexto---
{context}
---fin contexto---

Eres parte de una conversación continua, por lo que puedes hacer referencia a preguntas anteriores si es relevante.

Si encuentras la información en el contexto, respóndela con seguridad y de forma clara. Si no la tienes, responde brevemente con educación, pero sin explicar por qué ni mencionar ninguna fuente, contexto, documento o limitación.

No digas frases como 'no tengo información suficiente', 'basado en el documento' o similares. Solo responde lo que sabes con elegancia. Si no sabes, di algo educado y neutro.

No menciones contexto, datos disponibles ni restricciones. No expliques por qué puedes o no puedes responder. Solo responde o responde con cortesía, sin justificar.

En definitiva, si la respuesta está en el contexto la das, sino no, pero sin nunca mencionar tus restricciones.

Y varía un poco tus respuestas negativas, no digas siempre lo mismo.
"""

def generate_answer(messages):
    """Genera respuesta usando OpenAI GPT con historial completo"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generando respuesta: {e}")
        return "Lo siento, ha ocurrido un error al procesar tu pregunta. Por favor, inténtalo de nuevo."

def get_or_create_session_id():
    """Obtiene o crea un ID de sesión único"""
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
    return session['chat_session_id']

def get_conversation_history(session_id):
    """Obtiene el historial de conversación para una sesión"""
    if session_id not in conversations:
        conversations[session_id] = []
    return conversations[session_id]

@chatbot_pedidos_bp.route('/')
def index_route():
    """Página principal del chatbot"""
    # Cargar datos si no están cargados
    if index is None or chunks is None:
        load_chatbot_data()
    
    return render_template('chatbot_pedidos.html')

@chatbot_pedidos_bp.route('/chat', methods=['POST'])
def chat():
    """Endpoint para procesar mensajes del chat"""
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        
        if not question:
            return jsonify({'error': 'Mensaje vacío'}), 400
        
        # Asegurar que los datos están cargados
        if index is None or chunks is None:
            if not load_chatbot_data():
                return jsonify({'error': 'Error del sistema. Los datos del chatbot no están disponibles.'}), 500
        
        # Obtener o crear sesión
        session_id = get_or_create_session_id()
        messages = get_conversation_history(session_id)
        
        # Buscar chunks relevantes
        relevant_chunks = search_similar_chunks(question)
        context = "\n\n".join(relevant_chunks)
        
        # Si es el primer mensaje de la sesión, añadir system prompt
        if not messages:
            system_prompt = get_system_prompt(context)
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Actualizar el contexto en el system prompt existente (opcional)
            # Para mantener el contexto siempre fresco con la nueva búsqueda
            messages[0]["content"] = get_system_prompt(context)
        
        # Añadir el mensaje del usuario al historial
        messages.append({"role": "user", "content": question})
        
        # Generar respuesta
        answer = generate_answer(messages)
        
        # Añadir la respuesta del asistente al historial
        messages.append({"role": "assistant", "content": answer})
        
        # Actualizar el historial en la sesión
        conversations[session_id] = messages
        
        return jsonify({'response': answer})
        
    except Exception as e:
        print(f"Error en /chat: {e}")
        return jsonify({'error': 'Ha ocurrido un error interno. Por favor, inténtalo de nuevo.'}), 500

@chatbot_pedidos_bp.route('/clear-chat', methods=['POST'])
def clear_chat():
    """Endpoint para limpiar el historial de chat"""
    try:
        session_id = get_or_create_session_id()
        if session_id in conversations:
            del conversations[session_id]
        # También limpiar la sesión
        session.pop('chat_session_id', None)
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error limpiando chat: {e}")
        return jsonify({'error': 'Error al limpiar el chat'}), 500