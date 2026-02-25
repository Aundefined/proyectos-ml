document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const clearChatBtn = document.getElementById('clearChatBtn');

    // Auto-resize del textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = this.scrollHeight + 'px';
        sendButton.disabled = this.value.trim() === '';
    });

    // Envío con Enter (sin Shift)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendButton.addEventListener('click', sendMessage);

    clearChatBtn.addEventListener('click', function() {
        if (confirm('¿Quieres iniciar una nueva conversación? El historial actual se perderá.')) {
            clearChat();
        }
    });

    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message || sendButton.disabled) return;

        addMessage(message, 'user');
        messageInput.value = '';
        messageInput.style.height = 'auto';
        sendButton.disabled = true;

        showTypingIndicator();

        fetch('/chatbot-gemelo/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            if (data.error) {
                addMessage('Lo siento, ha ocurrido un error: ' + data.error, 'bot');
            } else {
                addMessage(data.response, 'bot');
            }
        })
        .catch(error => {
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
        avatar.innerHTML = sender === 'user'
            ? '<i class="bi bi-person-circle"></i>'
            : `<img src="${GEMELO_AVATAR_URL}" alt="Arnaud">`;

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
        let formatted = text.replace(/\n/g, '<br>');
        formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
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
        avatar.innerHTML = `<img src="${GEMELO_AVATAR_URL}" alt="Arnaud">`;

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
        if (indicator) indicator.remove();
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    function clearChat() {
        fetch('/chatbot-gemelo/clear-chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Eliminar todos los mensajes salvo el de bienvenida (primero)
                const messages = chatMessages.querySelectorAll('.message');
                messages.forEach((message, index) => {
                    if (index > 0) message.remove();
                });
            } else {
                alert('Error al limpiar el chat. Por favor, recarga la página.');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error de conexión al limpiar el chat. Por favor, recarga la página.');
        });
    }

    messageInput.focus();
});
