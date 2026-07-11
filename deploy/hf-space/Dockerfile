# PlastiScope — Hugging Face Space (Docker SDK)
# Builds by cloning the public GitHub repo (app code + trained model),
# then serves the FastAPI app on the port Hugging Face expects (7860).

FROM python:3.11-slim

# git is only needed to pull the project at build time
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone the project — the model (models/microplastic_yolo26m.onnx) comes with it.
# --depth 1 keeps the build fast (no history).
RUN git clone --depth 1 https://github.com/Saadman80/AI-Based-Microplastic-Detection-System.git .

# Install only the lean web stack (fastapi, uvicorn, onnxruntime, pillow, numpy, python-multipart)
RUN pip install --no-cache-dir -r requirements.txt

# server.py resolves the model as <repo>/models/...  and static as webapp/static
WORKDIR /app/webapp

EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
