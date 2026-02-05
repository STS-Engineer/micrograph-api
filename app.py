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

import requests
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from openai import OpenAI
from werkzeug.utils import secure_filename

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

# Dossier pour le stockage temporaire sur le serveur
TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Upload limits (16MB)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

INPUT_PPT_DIR.mkdir(parents=True, exist_ok=True)

# Initialisation du client OpenAI (API key via env OPENAI_API_KEY)
client = OpenAI()

# -----------------------------
# TEMP UPLOAD VALIDATION (like your 1st code)
# -----------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf", "txt", "csv", "xlsx", "docx", "pptx", "md", "json"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def guess_extension_from_mime(mime_type: str | None) -> str | None:
    if not mime_type:
        return None
    mt = mime_type.lower()
    if "png" in mt:
        return ".png"
    if "jpeg" in mt or "jpg" in mt:
        return ".jpg"
    if "pdf" in mt:
        return ".pdf"
    if "json" in mt:
        return ".json"
    if "csv" in mt:
        return ".csv"
    if "text" in mt or "plain" in mt:
        return ".txt"
    if "word" in mt or "docx" in mt:
        return ".docx"
    if "presentation" in mt or "pptx" in mt:
        return ".pptx"
    if "spreadsheet" in mt or "xlsx" in mt:
        return ".xlsx"
    if "markdown" in mt:
        return ".md"
    return None

# -----------------------------
# BACKGROUND CLEANUP TASK
# -----------------------------
def cleanup_old_files(interval=1800):  # toutes les 30 minutes
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
    except Exception:
        return jsonify({"error": "not_found"}), 404

@app.route("/temp_files/<path:filename>", methods=["GET"])
def serve_temp_image(filename):
    """Sert les fichiers temporaires stockés localement"""
    try:
        return send_from_directory(str(TEMP_UPLOAD_DIR), filename)
    except Exception:
        return jsonify({"error": "temp_file_not_found"}), 404

# -----------------------------
# LOCAL TEMPORARY STORAGE (UPDATED)
# -----------------------------
@app.route("/upload_temp_image", methods=["POST"])
def upload_temp_image():
    """
    Logique type "premier code" MAIS stockage temporaire local:
    - Reçoit openaiFileIdRefs: [ {id, download_link?, name?, mime_type?}, ... ] ou ["file-..."]
    - (Compat) accepte aussi {"file_id": "..."} (ancien format)
    - Télécharge bytes: download_link -> fallback OpenAI file_id
    - Valide extension (ALLOWED_EXTENSIONS)
    - Sauvegarde dans temp_uploads/
    - Retourne URLs locales /temp_files/<filename>
    """
    data = request.get_json(silent=True) or {}

    refs = data.get("openaiFileIdRefs")

    # Backward-compat: ancien payload {"file_id": "..."}
    if not refs and data.get("file_id"):
        refs = [{"id": data["file_id"], "name": None, "download_link": None, "mime_type": None}]

    if not refs or not isinstance(refs, list):
        return jsonify(
            {
                "success": False,
                "error": "missing_openaiFileIdRefs",
                "message": "Provide openaiFileIdRefs (list) or legacy file_id",
            }
        ), 400

    uploaded_results = []
    errors = []

    for file_ref in refs:
        try:
            # Normalize input
            if isinstance(file_ref, dict):
                file_id = file_ref.get("id")
                download_link = file_ref.get("download_link")
                original_name = file_ref.get("name") or "uploaded_file"
                mime_type = file_ref.get("mime_type")
            else:
                file_id = str(file_ref)
                download_link = None
                original_name = "uploaded_file"
                mime_type = None

            if not file_id:
                errors.append("Missing file_id in file reference")
                continue

            # Download bytes (LINK -> FILE_ID fallback)
            file_bytes = None

            if download_link:
                try:
                    print(f"⬇️ Trying download_link for {original_name}")
                    r = requests.get(download_link, timeout=15)
                    r.raise_for_status()
                    file_bytes = r.content
                except Exception as e:
                    print(f"⚠️ download_link failed, falling back to file_id: {e}")

            if file_bytes is None:
                # If name missing, try to retrieve filename from OpenAI
                if not original_name or original_name == "uploaded_file":
                    try:
                        file_info = client.files.retrieve(file_id)
                        if getattr(file_info, "filename", None):
                            original_name = file_info.filename
                        if not mime_type and getattr(file_info, "purpose", None):
                            # purpose isn't mime; keep mime_type as-is
                            pass
                    except Exception:
                        pass

                file_bytes = client.files.content(file_id).read()

            # Filename sanitization + extension handling
            filename_safe = secure_filename(original_name or "uploaded_file")

            if "." not in filename_safe:
                ext = guess_extension_from_mime(mime_type) or ".bin"
                filename_safe += ext

            if not allowed_file(filename_safe):
                errors.append(f"{original_name}: File type not allowed")
                continue

            # Save locally with unique name
            unique_filename = f"{uuid.uuid4().hex}_{int(time.time())}_{filename_safe}"
            file_path = TEMP_UPLOAD_DIR / unique_filename

            with open(file_path, "wb") as f:
                f.write(file_bytes)

            file_url = f"{request.host_url.rstrip('/')}/temp_files/{unique_filename}"

            uploaded_results.append(
                {
                    "original_name": original_name,
                    "filename": unique_filename,
                    "url": file_url,
                    "expires_in": "2 hours",
                }
            )

        except Exception as e:
            print(f"❌ Error processing {file_ref}: {e}")
            errors.append(f"{file_ref}: {str(e)}")

    if not uploaded_results and errors:
        return jsonify({"success": False, "message": "All uploads failed", "errors": errors}), 500

    return jsonify(
        {
            "success": True,
            "message": f"Processed {len(uploaded_results)} files.",
            "files": uploaded_results,
            "errors": errors,
        }
    ), 200

# -----------------------------
# SEARCH
# -----------------------------
@app.route("/search", methods=["POST"])
def search():
    """
    Search for similar micrographs.
    Accepts either:
      - temp_filename (from /upload_temp_image)
      - file_id (OpenAI)
    """
    if ENGINE is None:
        return jsonify({"error": "engine_not_loaded"}), 500

    data = request.get_json(silent=True) or {}
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
            if getattr(file_info, "purpose", None) not in ["assistants", "vision", "assistants_output"]:
                return jsonify(
                    {
                        "error": "invalid_file_purpose",
                        "message": f"File purpose is '{getattr(file_info, 'purpose', None)}'. Must be 'assistants' or 'vision'. Please re-upload the file with correct purpose.",
                    }
                ), 400

            # Télécharger le contenu du fichier depuis OpenAI
            file_content = client.files.content(file_id).read()
            img = Image.open(io.BytesIO(file_content)).convert("RGB")

        except Exception as e:
            error_msg = str(e)
            if "No such File object" in error_msg or "Could not find" in error_msg:
                return jsonify(
                    {
                        "error": "file_not_accessible",
                        "message": "The file_id cannot be accessed. This usually means: (1) The file is a conversation attachment, not uploaded via Files API, or (2) The file has expired. Please upload the image using OpenAI's Files API with purpose='assistants'.",
                        "hint": "In GPT, use the file upload function with purpose='assistants' before calling this API.",
                    }
                ), 400
            return jsonify({"error": "openai_retrieval_failed", "message": error_msg}), 400

    else:
        return jsonify({"error": "missing_input", "message": "Provide either file_id or temp_filename"}), 400

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
        build_faiss_index(
            embeddings_path=embeddings_path, output_dir=str(OUTPUT_BASE_DIR), model_name=MODEL_NAME
        )

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
