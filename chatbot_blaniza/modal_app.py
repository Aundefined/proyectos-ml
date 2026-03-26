import modal

app = modal.App("blaniza-assistant")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers>=4.37.0",
        "torch>=2.1.0",
        "accelerate>=0.26.0",
        "huggingface_hub>=0.20.0",
        "psutil",
        "fastapi[standard]",
    )
)

MODEL_NAME = "ArnaudClaudeML/ASISTENTE-SFTT-Qwen2p5-7B-LoRA-merged"
MODEL_DIR = "/model"

# Volumen persistente para cachear el modelo
volume = modal.Volume.from_name("blaniza-model-cache", create_if_missing=True)


def download_model():
    """Descarga el modelo al volumen. Solo se ejecuta una vez."""
    from huggingface_hub import snapshot_download
    import os

    print(f"Descargando modelo {MODEL_NAME} en {MODEL_DIR}...")
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_DIR,
    )
    print("Modelo descargado correctamente.")


# Imagen con el modelo ya descargado en el volumen
image_with_model = image.run_function(
    download_model,
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={MODEL_DIR: volume},
)


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "healthy", "model": MODEL_NAME}


@app.cls(
    image=image_with_model,
    gpu="T4",
    secrets=[modal.Secret.from_name("hf-secret")],
    volumes={MODEL_DIR: volume},
    scaledown_window=300,  # escala a cero tras 5 min sin peticiones
)
class BlanizaAssistant:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Cargando modelo desde {MODEL_DIR}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Dispositivo: {device}, dtype: {dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=dtype,
            device_map="auto",
            use_safetensors=True,
            trust_remote_code=True,
        )
        self.model.eval()
        print("Modelo cargado correctamente.")

    @modal.fastapi_endpoint(method="POST")
    def chat(self, request: dict):
        import torch
        import gc

        try:
            messages = request.get("messages", [])
            max_tokens = request.get("max_tokens", 200)

            if not messages:
                return {"error": "No se recibieron mensajes"}

            # Limitar historial y truncar mensajes largos (igual que en el Space)
            recent_messages = messages[-3:] if len(messages) > 3 else messages
            trimmed_messages = []
            for message in recent_messages:
                content = message.get("content", "")
                if isinstance(content, str) and len(content) > 1000:
                    content = content[:1000] + "..."
                trimmed_messages.append({
                    "role": message.get("role", "user"),
                    "content": content
                })

            prompt = self.tokenizer.apply_chat_template(
                trimmed_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            generation_params = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "temperature": 0.3,
                "top_p": 1.0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            with torch.no_grad():
                try:
                    outputs = self.model.generate(inputs.input_ids, **generation_params)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        generation_params["max_new_tokens"] = 30
                        gc.collect()
                        torch.cuda.empty_cache()
                        outputs = self.model.generate(inputs.input_ids, **generation_params)
                    else:
                        raise e

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            ).strip()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {"response": response, "status": "success"}

        except Exception as e:
            print(f"Error generando respuesta: {e}")
            return {"error": str(e)}

