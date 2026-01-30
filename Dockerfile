# Base Image: Using slim variant to minimize attack surface and image size
FROM python:3.10-slim-bullseye

# Metadata for image registry management
LABEL maintainer="DocuMind Engineering"
LABEL description="Production environment for RAG Agentic Workspace"

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
# PYTHONUNBUFFERED: Ensures logs are flushed directly to stdout (essential for Docker logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# System Dependencies
# Installing only essential build tools and cleaning up apt cache to reduce layer size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dependency Layer Caching
# Copying requirements first optimizes build time by leveraging Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Application Code
COPY . .

# Security: Create and switch to a non-root user
# Running as root is a security risk; creating a dedicated 'appuser'
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Orchestration Configuration
EXPOSE 8501

# Healthcheck: Liveness probe for container orchestrators (K8s/ECS)
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Runtime Entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
