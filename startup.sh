#!/bin/bash
set -e

echo "🚀 Azure Deployment - Micrograph Search Engine"

# --- 1. Environment Setup ---
export PYTHONUNBUFFERED=1
export TRANSFORMERS_CACHE=/tmp/transformers_cache
mkdir -p $TRANSFORMERS_CACHE

# --- 2. Dependency Installation ---
echo "📦 Installing dependencies..."
pip install --upgrade pip --no-cache-dir
pip install -r requirements.txt --no-cache-dir

# --- 3. Model Pre-caching (Optional - only if you want immediate availability) ---
# This downloads ~1.2GB model, comment out for faster cold starts
if [ "${PRECACHE_MODEL:-false}" = "true" ]; then
    echo "🧬 Pre-caching DINOv2-Large model..."
    python3 -c "from transformers import AutoModel, AutoImageProcessor; \
                AutoModel.from_pretrained('facebook/dinov2-large'); \
                AutoImageProcessor.from_pretrained('facebook/dinov2-large')"
else
    echo "⚡ Skipping model pre-cache for faster startup (lazy loading enabled)"
fi

# --- 4. Launch Flask API ---
echo "🚀 Starting API server on port ${PORT:-8000}..."
exec gunicorn app:app \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers ${WORKERS:-2} \
    --timeout ${TIMEOUT:-300} \
    --access-logfile - \
    --error-logfile -
