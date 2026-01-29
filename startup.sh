#!/bin/bash

# --- 1. Environment Setup ---
echo "🌐 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# --- 2. Dependency Installation ---
echo "📦 Installing core dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# --- 4. Model Pre-caching ---
# Force download the DINOv2 model to avoid timeout on first API call
echo "🧬 Pre-caching DINOv2-Large model..."
python3 -c "from transformers import AutoModel, AutoImageProcessor; \
            AutoModel.from_pretrained('facebook/dinov2-large'); \
            AutoImageProcessor.from_pretrained('facebook/dinov2-large')"

# --- 5. Launch the Flask API ---
echo "🚀 Starting Micrograph Search Engine on port 8000..."
python app.py --host 0.0.0.0 --port 8000
