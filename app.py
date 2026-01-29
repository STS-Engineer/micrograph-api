import os
import argparse
import shutil
import json
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image
import torch

# Imports from your local scripts
# NOTE: Ensure the file is renamed to extract_references_french_v3_3.py
from extract_references_french_v3_3 import process_powerpoint
from compute_embeddings import EmbeddingComputer, save_embeddings
from build_faiss_index_proper import build_faiss_index
from search_similar_french_v2 import FrenchMicrographSearchEngine

app = Flask(__name__)

# --- CONFIGURATION ---
ENGINE = None
INPUT_PPT_DIR = "./inputs"
OUTPUT_BASE_DIR = "./embeddings_v7"
MODEL_NAME = "dinov2"

def load_engine(config_path: str):
    """Loads or reloads the search engine into memory."""
    global ENGINE
    print(f"🔄 Loading search engine with config: {config_path}")
    ENGINE = FrenchMicrographSearchEngine(config_path=config_path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "engine_loaded": ENGINE is not None,
        "model": MODEL_NAME
    }), 200

@app.route("/update_index", methods=["POST"])
def update_index():
    """
    Pipeline Route:
    1. Purge old output folder.
    2. Extract images/metadata from PPTs in /inputs.
    3. Compute DINOv2 embeddings.
    4. Build FAISS index.
    5. Refresh the Search Engine.
    """
    global ENGINE
    try:
        # 1. Reset Output Directory
        if os.path.exists(OUTPUT_BASE_DIR):
            print(f"🧹 Deleting old index at {OUTPUT_BASE_DIR}")
            shutil.rmtree(OUTPUT_BASE_DIR)
        os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

        # 2. Extract Data from PowerPoint
        input_path = Path(INPUT_PPT_DIR)
        ppt_files = list(input_path.glob("*.pptx")) + list(input_path.glob("*.ppt"))
        
        if not ppt_files:
            return jsonify({"error": "no_ppt_files_found", "path": str(input_path.resolve())}), 400

        print("📸 Step 1: Extracting images and metadata...")
        all_metadata = []
        for ppt_file in ppt_files:
            # Uses logic from extract_references_french_v3_3.py
            metadata = process_powerpoint(ppt_file, Path(OUTPUT_BASE_DIR))
            all_metadata.extend(metadata)

        # Save interim metadata for the computer
        temp_meta_path = os.path.join(OUTPUT_BASE_DIR, "metadata.json")
        with open(temp_meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        # 3. Compute Embeddings
        print("🧬 Step 2: Computing embeddings...")
        computer = EmbeddingComputer(model_name=MODEL_NAME) #
        embeddings, valid_metadata = computer.compute_batch(
            all_metadata, 
            metadata_path=temp_meta_path, 
            images_root=os.path.join(OUTPUT_BASE_DIR, "images")
        )
        save_embeddings(embeddings, valid_metadata, OUTPUT_BASE_DIR, MODEL_NAME)

        # 4. Build FAISS Index
        print("🏗️ Step 3: Building FAISS index...")
        embeddings_path = os.path.join(OUTPUT_BASE_DIR, f"embeddings_{MODEL_NAME}.npy")
        build_faiss_index( #
            embeddings_path=embeddings_path,
            output_dir=OUTPUT_BASE_DIR,
            model_name=MODEL_NAME
        )

        # 5. Hot-Reload Engine
        new_config = os.path.join(OUTPUT_BASE_DIR, f"search_config_{MODEL_NAME}.json")
        load_engine(new_config)

        # Optional: Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({
            "status": "success",
            "images_indexed": len(valid_metadata),
            "config": new_config
        }), 200

    except Exception as e:
        print(f"❌ Update failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/search", methods=["POST"])
def search():
    """Similarity search using the current loaded engine."""
    if ENGINE is None:
        return jsonify({"error": "engine_not_loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "missing_file"}), 400

    f = request.files["file"]
    try:
        img = Image.open(f.stream).convert("RGB")
        # Perform search using search_similar_french_v2.py logic
        results = ENGINE.search_from_pil(img, top_k=1)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": "search_failed", "message": str(e)}), 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Ensure input directory exists
    os.makedirs(INPUT_PPT_DIR, exist_ok=True)

    # Auto-load existing index if it exists
    existing_config = os.path.join(OUTPUT_BASE_DIR, f"search_config_{MODEL_NAME}.json")
    if os.path.exists(existing_config):
        try:
            load_engine(existing_config)
        except Exception as e:
            print(f"⚠️ Could not auto-load existing index: {e}")

    app.run(host=args.host, port=args.port)