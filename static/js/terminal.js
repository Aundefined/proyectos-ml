(function () {
    const CHAT_URL  = '/sechat/chat';
    const CLEAR_URL = '/sechat/clear';

    let history    = [];
    let isThinking = false;

    function getOutput() { return document.getElementById('llm-output'); }
    function getInput()  { return document.getElementById('llm-input'); }

    function appendLine(text, cssClass) {
        const div = document.createElement('div');
        div.className = cssClass;
        div.textContent = text;
        getOutput().appendChild(div);
        getOutput().scrollTop = getOutput().scrollHeight;
        return div;
    }

    function removeThinking() {
        const el = getOutput().querySelector('.term-thinking');
        if (el) el.remove();
    }

    async function sendMessage() {
        if (isThinking) return;
        const input   = getInput();
        const message = input.value.trim();
        if (!message) return;

        input.value = '';
        appendLine(message, 'term-user');
        appendLine('pensando...', 'term-thinking');
        isThinking = true;

        try {
            const res  = await fetch(CHAT_URL, {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify({ message, history }),
            });
            const data = await res.json();
            removeThinking();

            if (data.error) {
                appendLine(data.error, 'term-error');
            } else {
                appendLine(data.response, 'term-bot');
                history.push({ role: 'user',      content: message });
                history.push({ role: 'assistant',  content: data.response });
                if (history.length > 4) history = history.slice(-4);
            }
        } catch (_) {
            removeThinking();
            appendLine('Error de conexión. Inténtalo de nuevo.', 'term-error');
        } finally {
            isThinking = false;
            input.focus();
        }
    }

    async function clearChat() {
        try { await fetch(CLEAR_URL, { method: 'POST' }); } catch (_) {}
        history = [];
        const output = getOutput();
        while (output.children.length > 1) output.removeChild(output.lastChild);
        getInput().focus();
    }

    document.addEventListener('DOMContentLoaded', function () {
        const modal = document.getElementById('terminalLLMModal');
        if (!modal) return;

        getInput().addEventListener('keydown', function (e) {
            if (e.key === 'Enter') sendMessage();
            if (e.key === 'l' && e.ctrlKey) { e.preventDefault(); clearChat(); }
        });

        document.getElementById('llm-send-btn').addEventListener('click', sendMessage);
        document.getElementById('llm-clear-btn').addEventListener('click', clearChat);

        modal.addEventListener('shown.bs.modal',  () => getInput().focus());
        modal.addEventListener('hidden.bs.modal', () => clearChat());
    });
})();
