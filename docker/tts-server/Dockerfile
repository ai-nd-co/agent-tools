# agent-tools tts-server as a container: Silero v5 Russian on CPU, OpenAI speech API, one Bearer key.
#
# Built for the office speech pod (k3s-gpu-cluster, namespace speech), where it sits next to a
# Speaches container that owns English (Kokoro) and speech-to-text. It needs no GPU — Silero
# synthesizes a sentence in ~0.3 s on two CPU cores — so the pod gives it NVIDIA_VISIBLE_DEVICES=void.
#
# The model is baked in at build time, verified by size and SHA-256 by agent-tools itself, so a
# container start needs no network and no volume. Silero's own licence (CC BY-NC-SA 4.0) covers the
# model; this image is for the owner's tooling, not for sale.
#
# Run:  docker run -e API_KEY=<64 hex> -p 8081:8081 ghcr.io/ai-nd-co/agent-tools-tts-server:<tag>
#       curl -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
#            -d '{"model":"silero-v5-ru","voice":"eugene","input":"Готово.","response_format":"wav"}' \
#            http://127.0.0.1:8081/v1/audio/speech-ru --output ready.wav

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AGENT_TOOLS_SILERO_CACHE_DIR=/models/silero \
    OMP_NUM_THREADS=2

RUN useradd --create-home --uid 1000 app && mkdir -p /models/silero && chown -R app:app /models

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# CPU torch from PyTorch's own index (the PyPI wheel drags CUDA libraries in); the package itself
# without its dependency list, because that list includes Kokoro and its stack, which this image
# never runs. What remains is exactly what tts_server, ttsify and silero_tts import.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" \
    && pip install --no-deps . \
    && pip install "numpy>=1.26.0" "httpx>=0.28.1" "cryptography>=44.0.0" "qrcode>=8.0"

# The model, once, into the image. ensure_silero_model() refuses a wrong size or digest.
USER app
RUN python -c "from agent_tools.silero_tts import ensure_silero_model; print(ensure_silero_model())"

EXPOSE 8081
# Container form of the server: wildcard bind (the pod network is the boundary), token from the
# environment (how a Kubernetes Secret arrives), Silero only, CPU.
ENTRYPOINT ["agent-tools", "tts-server", "--host", "0.0.0.0", "--bind-any", "--port", "8081", \
            "--token-env", "API_KEY", "--engines", "silero", "--device", "cpu"]
