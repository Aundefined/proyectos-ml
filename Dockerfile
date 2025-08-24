FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias base
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo (esto se hace en build time)
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('ArnaudClaudeML/blaniza-assistant'); \
    AutoModelForCausalLM.from_pretrained('ArnaudClaudeML/blaniza-assistant', torch_dtype='auto', low_cpu_mem_usage=True)"

# Copiar código de la aplicación
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]