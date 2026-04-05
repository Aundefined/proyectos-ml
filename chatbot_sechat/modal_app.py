"""
modal_app.py

Sirve el LLM entrenado desde cero (ONNX) en Modal.

Antes del primer deploy:
    1. Exportar el modelo:
       cd "llm-desde-cero" && python export_onnx.py

    2. Crear el volumen y subir los archivos:
       modal volume create sechat-cache
       modal volume put sechat-cache checkpoints/instruct_model.onnx /model/instruct_model.onnx
       modal volume put sechat-cache data/vocab.json /model/vocab.json

    3. Desplegar (desde esta carpeta):
       modal deploy modal_app.py

    4. Copiar las URLs generadas a las variables de entorno de Railway:
       SECHAT_CHAT_URL   = <URL del endpoint /chat>
       SECHAT_HEALTH_URL = <URL del endpoint /health>
"""

import json
import modal

app = modal.App("sechat")

MODEL_DIR = "/model"
volume = modal.Volume.from_name("sechat-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime>=1.18.0",
        "tiktoken",
        "numpy",
        "fastapi[standard]",
    )
)

# Parámetros de generación (deben coincidir con chat_sft.py)
GEN_CONFIG = {
    "max_new_tokens"    : 100,
    "temperature"       : 0.2,
    "top_k"             : 20,
    "repetition_penalty": 1.15,
    "block_size"        : 1024,
}


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "healthy"}


@app.cls(
    image=image,
    volumes={MODEL_DIR: volume},
    scaledown_window=300,   # escala a cero tras 5 min sin peticiones
)
class LLMPropio:

    @modal.enter()
    def load(self):
        import onnxruntime as ort
        import tiktoken
        import numpy as np
        import os

        print("Cargando tokenizador...")
        with open(os.path.join(MODEL_DIR, "vocab.json"), "r") as f:
            vocab_data = json.load(f)

        enc = tiktoken.get_encoding(vocab_data["encoding_name"])
        t2c = {int(k): v for k, v in vocab_data["tiktoken_to_compact"].items()}
        c2t = {int(k): v for k, v in vocab_data["compact_to_tiktoken"].items()}

        self.encode = lambda text: [t2c[t] for t in enc.encode(text, allowed_special="all") if t in t2c]
        self.decode = lambda ids:  enc.decode([c2t[c] for c in ids if c in c2t])

        print("Cargando modelo ONNX...")
        onnx_path = os.path.join(MODEL_DIR, "instruct_model.onnx")
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        print("Modelo listo.")

    @modal.fastapi_endpoint(method="POST")
    def chat(self, request: dict):
        import numpy as np

        message = (request.get("message") or "").strip()
        history = request.get("history") or []

        if not message:
            return {"error": "Mensaje vacío"}

        # Construir prompt con historial (máx 4 intercambios)
        history = history[-4:]
        prompt = ""
        for msg in history:
            if msg["role"] == "user":
                prompt += f"### Instrucción:\n{msg['content']}\n### Respuesta:\n"
            else:
                prompt += f"{msg['content']}\n###\n"
        prompt += f"### Instrucción:\n{message}\n### Respuesta:\n"

        try:
            ids = self.encode(prompt)
            block_size = GEN_CONFIG["block_size"]
            temperature = GEN_CONFIG["temperature"]
            top_k = GEN_CONFIG["top_k"]
            rep = GEN_CONFIG["repetition_penalty"]

            for _ in range(GEN_CONFIG["max_new_tokens"]):
                ids_cond = ids[-block_size:]
                input_arr = np.array([ids_cond], dtype=np.int64)

                logits = self.session.run(["logits"], {"input_ids": input_arr})[0]
                logits = logits[0, -1, :].astype(np.float64)

                # Penalización por repetición
                if rep != 1.0:
                    for tid in set(ids):
                        if tid < len(logits):
                            if logits[tid] > 0:
                                logits[tid] /= rep
                            else:
                                logits[tid] *= rep

                logits /= temperature

                # Top-K sampling
                if top_k is not None:
                    k = min(top_k, len(logits))
                    threshold = np.sort(logits)[-k]
                    logits[logits < threshold] = -np.inf

                # Softmax + muestreo
                logits -= np.max(logits)
                probs = np.exp(logits)
                probs /= probs.sum()
                next_id = int(np.random.choice(len(probs), p=probs))
                ids.append(next_id)

                # Condición de parada
                generated = self.decode(ids)
                if "###" in generated.split("### Respuesta:\n")[-1]:
                    break

            full_text = self.decode(ids)
            response = full_text.split("### Respuesta:\n")[-1].replace("###", "").strip()
            return {"response": response}

        except Exception as e:
            print(f"Error generando respuesta: {e}")
            return {"error": str(e)}
