# Single image: React frontend + FastAPI backend + Hyderabad graph + trained models.
# Build on a Mac (Apple Silicon) for Render's x86 servers:
#   docker buildx build --platform linux/amd64 -t <dockerhub-user>/smartroute-ai:latest --push .

# ---------- 1. build the frontend ----------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
# empty VITE_API_URL -> the app calls the API on its own origin
ENV VITE_API_URL=
RUN npm run build

# ---------- 2. Python runtime ----------
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ENVIRONMENT=production \
    FRONTEND_DIST=/app/frontend/dist MODEL_CHECKPOINT_DIR=/app/reinforcement_learning/checkpoints
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt
COPY backend backend
COPY reinforcement_learning reinforcement_learning
COPY scripts scripts
COPY data/processed data/processed
COPY data/README.md data/README.md
COPY --from=web /web/dist frontend/dist
EXPOSE 10000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers"]
