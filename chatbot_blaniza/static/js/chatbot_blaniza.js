document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const healthBtn = document.getElementById('healthBtn');

    // Auto-resize del textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
        
        // Habilitar/deshabilitar botón de envío
        const isEmpty = this.value.trim() === '';
        sendButton.disabled = isEmpty;
    });

    // Envío con Enter (sin Shift)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Envío con botón
    sendButton.addEventListener('click', sendMessage);

    // Limpiar chat
    clearChatBtn.addEventListener('click', function() {
        if (confirm('¿Estás seguro de que quieres limpiar toda la conversación?')) {
            clearChat();
        }
    });

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message || sendButton.disabled) return;

        // Añadir mensaje del usuario
        addMessage(message, 'user');
        
        // Limpiar input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;

        // Mostrar indicador de escritura
        showTypingIndicator();

        // Enviar mensaje al servidor (chatbot-blaniza en lugar de chatbot-pedidos)
        fetch('/chatbot-blaniza/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Ocultar indicador de escritura
            hideTypingIndicator();
            
            // 🔍 TRAZA: Mostrar información completa de la respuesta
            console.group('🤖 RESPUESTA COMPLETA DEL CHATBOT');
            console.log('📦 Datos recibidos:', data);
            
            if (data.debug_info) {
                console.log('🔍 Debug del servidor:', data.debug_info);
                console.log('📊 Mensajes enviados al modelo:', data.debug_info.messages_sent || 'No disponible');
                console.log('📝 Historial completo:', data.debug_info.conversation_history || 'No disponible');
            }
            
            if (data.error) {
                console.error('❌ Error del servidor:', data.error);
                addMessage('Lo siento, ha ocurrido un error. Por favor, inténtalo de nuevo.', 'bot');
            } else {
                console.log('✅ Respuesta del modelo:', data.response);
                addMessage(data.response, 'bot');
            }
            console.groupEnd();
        })
        .catch(error => {
            // Ocultar indicador de escritura
            hideTypingIndicator();
            addMessage('Ha ocurrido un error de conexión. Por favor, inténtalo de nuevo.', 'bot');
            console.error('Error:', error);
        });
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = sender === 'user' ? '<i class="bi bi-person-circle"></i>' : '<i class="bi bi-robot"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = formatMessage(text);
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString('es-ES', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        content.appendChild(messageText);
        content.appendChild(messageTime);
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function formatMessage(text) {
        // Convertir saltos de línea a <br>
        let formatted = text.replace(/\n/g, '<br>');
        
        // Convertir texto en negrita (markdown style)
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Convertir listas simples
        formatted = formatted.replace(/^- (.+)$/gm, '<li>$1</li>');
        if (formatted.includes('<li>')) {
            formatted = formatted.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        }
        
        return formatted;
    }

    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.id = 'typing-indicator';
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="bi bi-robot"></i>';
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        
        content.appendChild(indicator);
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(content);
        
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
    }

    function hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    function clearChat() {
        // Llamar al endpoint para limpiar el historial del servidor (chatbot-blaniza)
        fetch('/chatbot-blaniza/clear-chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Limpiar la interfaz visual
                const messages = chatMessages.querySelectorAll('.message');
                messages.forEach((message, index) => {
                    // Mantener solo el primer mensaje de bienvenida
                    if (index > 0) {
                        message.remove();
                    }
                });
                console.log('Chat limpiado exitosamente');
            } else {
                console.error('Error limpiando el chat:', data.error);
                alert('Error al limpiar el chat. Por favor, recarga la página.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error de conexión al limpiar el chat. Por favor, recarga la página.');
        });
    }

    // Health check manual
    healthBtn.addEventListener('click', function() {
        healthBtn.disabled = true;
        healthBtn.className = 'btn btn-secondary';
        healthBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span> Comprobando...';

        fetch('/chatbot-blaniza/health')
            .then(response => response.json())
            .then(data => {
                if (data.available) {
                    healthBtn.className = 'btn btn-success';
                    healthBtn.innerHTML = '<i class="bi bi-circle-fill me-1"></i> Servicio disponible';
                } else {
                    healthBtn.className = 'btn btn-danger';
                    healthBtn.innerHTML = '<i class="bi bi-circle-fill me-1"></i> Servicio no disponible';
                }
            })
            .catch(() => {
                healthBtn.className = 'btn btn-danger';
                healthBtn.innerHTML = '<i class="bi bi-circle-fill me-1"></i> Servicio no disponible';
            })
            .finally(() => {
                healthBtn.disabled = false;
            });
    });

    // Focus inicial en el input
    messageInput.focus();
});