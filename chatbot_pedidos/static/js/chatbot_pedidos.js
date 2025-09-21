document.addEventListener('DOMContentLoaded', function() {
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatMessages = document.getElementById('chatMessages');
    const clearChatBtn = document.getElementById('clearChatBtn');
    const voiceButton = document.getElementById('voiceButton');
    const audioRecordingContainer = document.getElementById('audioRecordingContainer');
    const voiceHint = document.getElementById('voiceHint');

    // Variables para grabación de audio
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let recordingTimer = null;
    let recordingStartTime = null;
    const MAX_RECORDING_TIME = 20000; // 20 segundos

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

        // Enviar mensaje al servidor
        fetch('/chatbot-pedidos/chat', {
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
            
            if (data.error) {
                addMessage('Lo siento, ha ocurrido un error. Por favor, inténtalo de nuevo.', 'bot');
                console.error('Error:', data.error);
            } else {
                addMessage(data.response, 'bot');
            }
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

    function showTranscriptionIndicator() {
        const transcriptionDiv = document.createElement('div');
        transcriptionDiv.className = 'message bot-message';
        transcriptionDiv.id = 'transcription-indicator';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="bi bi-robot"></i>';

        const content = document.createElement('div');
        content.className = 'message-content';

        // Determinar qué modelo se está usando
        const selectedModel = document.getElementById('sttModelSelect').value;
        const modelText = selectedModel === 'openai' ? 'whisper-1' : 'whisper-large-v3';

        const indicator = document.createElement('div');
        indicator.className = 'transcription-indicator';
        indicator.innerHTML = `<i class="bi bi-mic-fill text-primary me-2"></i>Transcribiendo con ${modelText}<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;

        content.appendChild(indicator);
        transcriptionDiv.appendChild(avatar);
        transcriptionDiv.appendChild(content);

        chatMessages.appendChild(transcriptionDiv);
        scrollToBottom();
    }

    function hideTranscriptionIndicator() {
        const indicator = document.getElementById('transcription-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function showRecordingOverlay() {
        // Crear overlay de grabación
        const overlay = document.createElement('div');
        overlay.id = 'recording-overlay';
        overlay.className = 'recording-overlay';

        const content = document.createElement('div');
        content.className = 'recording-overlay-content';
        content.innerHTML = `
            <div class="recording-icon">
                <i class="bi bi-mic-fill"></i>
            </div>
            <div class="recording-text">GRABANDO</div>
            <div class="recording-timer-overlay">0s</div>
        `;

        overlay.appendChild(content);
        document.body.appendChild(overlay);
    }

    function hideRecordingOverlay() {
        const overlay = document.getElementById('recording-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    function scrollToBottom() {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }

    function clearChat() {
        // Llamar al endpoint para limpiar el historial del servidor
        fetch('/chatbot-pedidos/clear-chat', {
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

    // ==============================================
    // FUNCIONALIDAD DE GRABACIÓN DE VOZ (Push to Talk)
    // ==============================================

    // Configurar eventos para push-to-talk
    voiceButton.addEventListener('mousedown', handleVoiceButtonDown);
    voiceButton.addEventListener('mouseup', handleVoiceButtonUp);
    voiceButton.addEventListener('mouseleave', handleVoiceButtonUp); // Por si se sale del botón
    voiceButton.addEventListener('touchstart', handleVoiceButtonDown, { passive: false });
    voiceButton.addEventListener('touchend', handleVoiceButtonUp, { passive: false });

    // También manejar eventos globales para cuando el usuario suelta fuera del botón
    document.addEventListener('mouseup', handleVoiceButtonUp);
    document.addEventListener('touchend', handleVoiceButtonUp);

    function handleVoiceButtonDown(e) {
        e.preventDefault();
        e.stopPropagation();
        if (!isRecording) {
            startRecording();
        }
    }

    function handleVoiceButtonUp(e) {
        if (isRecording) {
            stopRecording();
        }
    }

    async function startRecording() {

        if (isRecording) return;

        try {
            // Solicitar permisos de micrófono
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });

            // Forzar WAV que es compatible con Whisper
            mediaRecorder = new MediaRecorder(stream);

            audioChunks = [];
            isRecording = true;
            recordingStartTime = Date.now();

            // Cambiar interfaz a modo grabación
            voiceButton.classList.add('recording');
            voiceButton.innerHTML = '<i class="bi bi-mic-fill text-danger"></i>';
            voiceHint.textContent = 'Mantén presionado para grabar...';

            // Mostrar overlay de grabación en el centro
            showRecordingOverlay();

            // Iniciar timer visual
            updateRecordingTimer();
            recordingTimer = setInterval(updateRecordingTimer, 100);

            // Configurar eventos del MediaRecorder
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = function() {
                processRecording();
            };

            // Iniciar grabación
            mediaRecorder.start();

            // Auto-stop después del tiempo máximo
            setTimeout(() => {
                if (isRecording) {
                    stopRecording();
                }
            }, MAX_RECORDING_TIME);

        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('No se pudo acceder al micrófono. Por favor, verifica los permisos.');
            resetRecordingState();
        }
    }

    function stopRecording() {
        if (!isRecording || !mediaRecorder) return;

        isRecording = false;
        clearInterval(recordingTimer);

        // Detener grabación
        if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
        }

        // Detener stream
        const stream = mediaRecorder.stream;
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }

        // Cambiar interfaz
        voiceButton.classList.remove('recording');
        voiceButton.innerHTML = '<i class="bi bi-mic-fill"></i>';
        voiceHint.textContent = 'Click en 🎤 para grabar audio (máx. 20s)';

        // Ocultar overlay de grabación
        hideRecordingOverlay();
    }

    function updateRecordingTimer() {
        if (!isRecording || !recordingStartTime) return;

        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);

        // Actualizar timer en la barra inferior (si existe)
        const timerElement = document.querySelector('.recording-timer');
        if (timerElement) {
            timerElement.textContent = `${elapsed}s`;
        }

        // Actualizar timer en el overlay
        const overlayTimer = document.querySelector('.recording-timer-overlay');
        if (overlayTimer) {
            overlayTimer.textContent = `${elapsed}s`;
        }
    }

    function resetRecordingState() {
        isRecording = false;
        clearInterval(recordingTimer);
        voiceButton.classList.remove('recording');
        voiceButton.innerHTML = '<i class="bi bi-mic-fill"></i>';
        voiceHint.textContent = 'Click en 🎤 para grabar audio (máx. 20s)';
        hideRecordingOverlay();
    }

    async function processRecording() {
        if (audioChunks.length === 0) {
            resetRecordingState();
            addMessage('No se detectó audio. Inténtalo de nuevo.', 'bot');
            return;
        }

        try {
            // Crear blob de audio
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

            // Verificar si hay contenido de audio real (no solo silencio)
            if (audioBlob.size < 8000) { // Archivo pequeño = probablemente solo silencio/ruido de fondo
                resetRecordingState();
                addMessage('No se detectó voz en la grabación. Asegúrate de hablar cerca del micrófono.', 'bot');
                return;
            }

            // Mostrar indicador de transcripción
            showTranscriptionIndicator();

            // Obtener modelo seleccionado
            const selectedModel = document.getElementById('sttModelSelect').value;

            // Enviar audio al servidor
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            formData.append('model', selectedModel);

            const response = await fetch('/chatbot-pedidos/transcribe-audio', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            // Ocultar indicador de transcripción
            hideTranscriptionIndicator();

            if (result.error) {
                console.error('Error transcribiendo:', result.error);

                // Diferenciar entre error de "no hay voz" y error técnico
                if (result.error.includes('No se detectó voz')) {
                    addMessage('No se detectó voz en la grabación. Asegúrate de hablar cerca del micrófono.', 'bot');
                } else {
                    addMessage('Error al procesar el audio. Por favor, inténtalo de nuevo.', 'bot');
                }
            } else {
                // Añadir mensaje transcrito del usuario
                addMessage(result.transcribed_text, 'user');
                // Añadir respuesta del bot
                addMessage(result.response, 'bot');
            }

        } catch (error) {
            console.error('Error processing recording:', error);
            // Ocultar indicador de transcripción en caso de error
            hideTranscriptionIndicator();
            addMessage('Error al enviar el audio. Por favor, inténtalo de nuevo.', 'bot');
        } finally {
            resetRecordingState();
        }
    }

    // Verificar soporte de MediaRecorder al cargar
    if (!navigator.mediaDevices || !MediaRecorder) {
        voiceButton.disabled = true;
        voiceButton.title = 'Grabación de voz no soportada en este navegador';
        voiceHint.textContent = 'Grabación de voz no disponible';
    }

    // Focus inicial en el input
    messageInput.focus();
});