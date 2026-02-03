import os
import argparse
import shutil
import json
import torch
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image

# Imports from your local scripts
from extract_references_french_v3_3 import process_powerpoint
from compute_embeddings import EmbeddingComputer, save_embeddings
from build_faiss_index_proper import build_faiss_index
from search_similar_french_v2 import FrenchMicrographSearchEngine

app = Flask(__name__)

# -----------------------------
# CONFIGURATION
# -----------------------------
ENGINE = None
INPUT_PPT_DIR = Path("inputs")
OUTPUT_BASE_DIR = Path("embeddings_v7")
IMAGES_DIR = OUTPUT_BASE_DIR / "images"
MODEL_NAME = "dinov2"

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

INPUT_PPT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# HELPERS
# -----------------------------
def load_engine(config_path: str):
    global ENGINE
    print(f"📄 Loading search engine: {config_path}")
    ENGINE = FrenchMicrographSearchEngine(config_path=config_path)


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "ok",
            "engine_loaded": ENGINE is not None,
            "model": MODEL_NAME,
        }
    ), 200


# -----------------------------
# IMAGE SERVING
# -----------------------------
@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_image(filename):
    """
    Serve images from the embeddings/images directory
    This handles the matched micrograph images
    """
    try:
        if IMAGES_DIR.exists():
            return send_from_directory(str(IMAGES_DIR), filename, as_attachment=False)
        else:
            return jsonify({"error": "images_directory_not_found"}), 404
    except Exception as e:
        return jsonify({"error": "image_not_found", "message": str(e)}), 404


# -----------------------------
# SEARCH
# -----------------------------
@app.route("/search", methods=["POST"])
def search():
    """
    Search for similar micrographs
    
    Request:
        - file: Image file (multipart/form-data)
        - top_k (optional): Number of results to return (default: 1)
    
    Response:
        {
            "results": [
                {
                    "similarity_score": 0.95,
                    "reference": "...",
                    "image_path": "...",
                    ...
                }
            ]
        }
    """
    if ENGINE is None:
        return jsonify({"error": "engine_not_loaded", "message": "Search engine is not initialized"}), 500

    if "file" not in request.files:
        return jsonify({"error": "missing_file", "message": "No file provided in request"}), 400

    f = request.files["file"]
    top_k = request.form.get("top_k", 1, type=int)

    try:
        img = Image.open(f.stream).convert("RGB")
        results = ENGINE.search_from_pil(img, top_k=top_k)
        return jsonify({"results": results}), 200

    except Exception as e:
        return jsonify({"error": "search_failed", "message": str(e)}), 400


# -----------------------------
# INDEX PIPELINE
# -----------------------------
@app.route("/update_index", methods=["POST"])
def update_index():
    """
    Rebuild the search index from PowerPoint files in the inputs directory
    
    This endpoint will:
    1. Extract images from all .ppt/.pptx files in the inputs directory
    2. Compute embeddings for all extracted images
    3. Build a FAISS index for fast similarity search
    4. Reload the search engine with the new index
    
    Response:
        {
            "status": "success",
            "images_indexed": 123
        }
    """
    global ENGINE

    try:
        # Clean output directory
        if OUTPUT_BASE_DIR.exists():
            shutil.rmtree(OUTPUT_BASE_DIR)
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

        # Find PowerPoint files
        ppt_files = list(INPUT_PPT_DIR.glob("*.pptx")) + list(INPUT_PPT_DIR.glob("*.ppt"))
        if not ppt_files:
            return jsonify({
                "error": "no_ppt_files_found",
                "message": f"No PowerPoint files found in {INPUT_PPT_DIR}"
            }), 400

        # Extract images and metadata from PowerPoints
        all_metadata = []
        for ppt_file in ppt_files:
            all_metadata.extend(process_powerpoint(ppt_file, OUTPUT_BASE_DIR))

        # Save metadata
        meta_path = OUTPUT_BASE_DIR / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        # Compute embeddings
        computer = EmbeddingComputer(model_name=MODEL_NAME)
        embeddings, valid_metadata = computer.compute_batch(
            all_metadata,
            metadata_path=str(meta_path),
            images_root=str(OUTPUT_BASE_DIR / "images"),
        )

        # Save embeddings
        save_embeddings(embeddings, valid_metadata, str(OUTPUT_BASE_DIR), MODEL_NAME)

        # Build FAISS index
        embeddings_path = str(OUTPUT_BASE_DIR / f"embeddings_{MODEL_NAME}.npy")
        build_faiss_index(
            embeddings_path=embeddings_path,
            output_dir=str(OUTPUT_BASE_DIR),
            model_name=MODEL_NAME,
        )

        # Reload search engine
        config_path = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
        load_engine(config_path)

        # Clean up GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify(
            {
                "status": "success",
                "images_indexed": len(valid_metadata),
            }
        ), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()

    # Try to load existing index on startup
    existing_config = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
    if os.path.exists(existing_config):
        try:
            load_engine(existing_config)
            print("✅ Search engine loaded successfully on startup")
        except Exception as e:
            print(f"⚠️ Engine auto-load failed: {e}")
            print("   Use /update_index endpoint to build a new index")

    print(f"\n🚀 API Server starting on {args.host}:{args.port}")
    print(f"   Health check: http://{args.host}:{args.port}/health")
    print(f"   Search: POST http://{args.host}:{args.port}/search")
    print(f"   Update index: POST http://{args.host}:{args.port}/update_index")
    
    app.run(host=args.host, port=args.port)
