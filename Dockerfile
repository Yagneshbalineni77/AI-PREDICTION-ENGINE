# ==========================================
# STAGE 1: Build Vue Frontend
# ==========================================
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend

# Install dependencies first for better caching
COPY frontend/package*.json ./
RUN npm ci

# Copy source and build
COPY frontend/ ./
# Add stub env file for build to prevent errors
RUN echo "VITE_API_BASE_URL=" > .env.production
RUN npm run build

# ==========================================
# STAGE 2: Production Environment
# ==========================================
FROM python:3.11-slim
WORKDIR /app

# Set environments
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    FLASK_ENV=production \
    FLASK_HOST=127.0.0.1 \
    FLASK_PORT=5001

# Install Nginx, Supervisord, and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (heavy dependency, cache it)
RUN pip install --no-cache-dir torch==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ backend/

# Set up Nginx
COPY nginx.conf /etc/nginx/nginx.conf
# Copy built Vue apps from stage 1
RUN rm -rf /usr/share/nginx/html/*
COPY --from=frontend-build /app/frontend/dist /usr/share/nginx/html

# Set up Supervisord
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Ensure critical directories exist
RUN mkdir -p /app/backend/uploads /app/backend/data /app/backend/logs

EXPOSE 3000

# Start supervisor to manage Nginx and Flask
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]