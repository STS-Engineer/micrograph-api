import os
import argparse
import shutil
import json
import torch
import time
import uuid
import io
from pathlib import Path
from threading import Thread

from flask import Flask, request, jsonify, send_from_directory, url_for
from PIL import Image
from openai import OpenAI

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

# Dossier pour le stockage temporaire sur le serveur Azure
TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

INPUT_PPT_DIR.mkdir(parents=True, exist_ok=True)

# Initialisation du client OpenAI
# L'API Key est récupérée automatiquement depuis la variable d'environnement OPENAI_API_KEY
client = OpenAI()


# -----------------------------
# BACKGROUND CLEANUP TASK
# -----------------------------
def cleanup_old_files(interval=1800): # Vérifie toutes les 30 minutes
    """
    Supprime les fichiers du dossier temp_uploads s'ils ont plus de 2 heures.
    """
    while True:
        now = time.time()
        for f in TEMP_UPLOAD_DIR.glob("*"):
            if f.is_file():
                # 7200 secondes = 2 heures
                if now - f.stat().st_mtime > 7200:
                    try:
                        f.unlink()
                        print(f"🗑️ Fichier temporaire supprimé : {f.name}")
                    except Exception as e:
                        print(f"⚠️ Erreur lors de la suppression de {f.name} : {e}")
        time.sleep(interval)

# Lancer le thread de nettoyage en arrière-plan
cleanup_thread = Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()


# -----------------------------
# HELPERS
# -----------------------------
def load_engine(config_path: str):
    global ENGINE
    print(f"📄 Loading search engine: {config_path}")
    ENGINE = FrenchMicrographSearchEngine(config_path=config_path)


# -----------------------------
# AUTO-LOAD ENGINE (works under Gunicorn)
# -----------------------------
existing_config = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
if os.path.exists(existing_config):
    try:
        load_engine(existing_config)
        print("✅ Search engine loaded on import (gunicorn)")
    except Exception as e:
        print(f"⚠️ Engine auto-load failed on import: {e}")
        print("   Use /update_index endpoint to build a new index")


# -----------------------------
# ROOT / HEALTH
# -----------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "status": "ok",
            "service": "micrographie-ia",
            "engine_loaded": ENGINE is not None,
            "model": MODEL_NAME,
            "endpoints": ["/health", "/search", "/upload_temp_image", "/temp_files/<filename>"],
        }
    ), 200


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
    """Serve images from the embeddings/images directory"""
    try:
        return send_from_directory(str(IMAGES_DIR), filename)
    except Exception as e:
        return jsonify({"error": "not_found"}), 404

@app.route("/temp_files/<path:filename>", methods=["GET"])
def serve_temp_image(filename):
    """Sert les images temporaires stockées localement"""
    try:
        return send_from_directory(str(TEMP_UPLOAD_DIR), filename)
    except Exception as e:
        return jsonify({"error": "temp_file_not_found"}), 404


# -----------------------------
# LOCAL TEMPORARY STORAGE
# -----------------------------
@app.route("/upload_temp_image", methods=["POST"])
def upload_temp_image():
    """
    Récupère une image depuis OpenAI via son file_id et la sauvegarde localement.
    Retourne l'URL locale pour que l'assistant GPT puisse l'utiliser.
    """
    data = request.get_json()
    if not data or "file_id" not in data:
        return jsonify({"error": "missing_file_id"}), 400

    file_id = data["file_id"]
    
    try:
        # Récupérer les infos du fichier pour avoir l'extension originale
        file_info = client.files.retrieve(file_id)
        ext = os.path.splitext(file_info.filename)[1] or ".png"
        
        # Télécharger le contenu du fichier
        file_content = client.files.content(file_id).read()
        
        # Générer un nom de fichier unique
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = TEMP_UPLOAD_DIR / unique_filename
        
        # Sauvegarder localement
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Construire l'URL complète
        file_url = f"{request.host_url.rstrip('/')}/temp_files/{unique_filename}"
        
        return jsonify({
            "url": file_url,
            "filename": unique_filename,
            "expires_in": "2 hours"
        }), 200
            
    except Exception as e:
        return jsonify({"error": "openai_retrieval_failed", "message": str(e)}), 500


# -----------------------------
# SEARCH
# -----------------------------
# Amélioration de l'endpoint /search dans app.py
# Remplacez votre fonction search() actuelle par celle-ci

@app.route("/search", methods=["POST"])
def search():
    """
    Search for similar micrographs
    Accepts either an OpenAI file_id or a filename from temp_uploads (via JSON)
    """
    if ENGINE is None:
        return jsonify({"error": "engine_not_loaded"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "missing_json_body"}), 400

    top_k = data.get("top_k", 1)
    img = None

    # Option 1: Recherche via un fichier déjà uploadé temporairement
    temp_filename = data.get("temp_filename")
    if temp_filename:
        file_path = TEMP_UPLOAD_DIR / temp_filename
        if file_path.exists():
            try:
                img = Image.open(file_path).convert("RGB")
            except Exception as e:
                return jsonify({"error": "invalid_temp_file", "message": str(e)}), 400
        else:
            return jsonify({"error": "temp_file_expired_or_not_found"}), 404

    # Option 2: Recherche via OpenAI file_id
    elif "file_id" in data:
        file_id = data["file_id"]
        try:
            # Vérifier d'abord que le fichier existe
            file_info = client.files.retrieve(file_id)
            
            # Vérifier le purpose (doit être "assistants" ou "vision")
            if file_info.purpose not in ["assistants", "vision", "assistants_output"]:
                return jsonify({
                    "error": "invalid_file_purpose",
                    "message": f"File purpose is '{file_info.purpose}'. Must be 'assistants' or 'vision'. Please re-upload the file with correct purpose."
                }), 400
            
            # Télécharger le contenu du fichier depuis OpenAI
            file_content = client.files.content(file_id).read()
            img = Image.open(io.BytesIO(file_content)).convert("RGB")
            
        except Exception as e:
            error_msg = str(e)
            
            # Message d'erreur plus explicite pour l'utilisateur GPT
            if "No such File object" in error_msg or "Could not find" in error_msg:
                return jsonify({
                    "error": "file_not_accessible",
                    "message": "The file_id cannot be accessed. This usually means: (1) The file is a conversation attachment, not uploaded via Files API, or (2) The file has expired. Please upload the image using OpenAI's Files API with purpose='assistants'.",
                    "hint": "In GPT, use the file upload function with purpose='assistants' before calling this API."
                }), 400
            else:
                return jsonify({
                    "error": "openai_retrieval_failed", 
                    "message": error_msg
                }), 400
    
    else:
        return jsonify({
            "error": "missing_input", 
            "message": "Provide either file_id or temp_filename"
        }), 400

    try:
        results = ENGINE.search_from_pil(img, top_k=top_k)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": "search_failed", "message": str(e)}), 400

# -----------------------------
# INDEX PIPELINE
# -----------------------------
@app.route("/update_index", methods=["POST"])
def update_index():
    """Rebuild the search index from PowerPoint files"""
    global ENGINE
    try:
        if OUTPUT_BASE_DIR.exists():
            shutil.rmtree(OUTPUT_BASE_DIR)
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

        ppt_files = list(INPUT_PPT_DIR.glob("*.pptx")) + list(INPUT_PPT_DIR.glob("*.ppt"))
        if not ppt_files:
            return jsonify({"error": "no_ppt_files_found"}), 400

        all_metadata = []
        for ppt_file in ppt_files:
            all_metadata.extend(process_powerpoint(ppt_file, OUTPUT_BASE_DIR))

        meta_path = OUTPUT_BASE_DIR / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        computer = EmbeddingComputer(model_name=MODEL_NAME)
        embeddings, valid_metadata = computer.compute_batch(
            all_metadata,
            metadata_path=str(meta_path),
            images_root=str(OUTPUT_BASE_DIR / "images"),
        )

        save_embeddings(embeddings, valid_metadata, str(OUTPUT_BASE_DIR), MODEL_NAME)

        embeddings_path = str(OUTPUT_BASE_DIR / f"embeddings_{MODEL_NAME}.npy")
        build_faiss_index(embeddings_path=embeddings_path, output_dir=str(OUTPUT_BASE_DIR), model_name=MODEL_NAME)

        config_path = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
        load_engine(config_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify({"status": "success", "images_indexed": len(valid_metadata)}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
