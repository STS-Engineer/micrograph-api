from __future__ import annotations

import argparse
import io
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import Optional

import requests
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI
from PIL import Image
from werkzeug.utils import secure_filename

# Imports from your local scripts (à adapter selon ton projet)
from extract_references_french_v3_3 import process_powerpoint  # noqa: F401

# --------------------------------------------------------------------------------------
# APP + PATHS
# --------------------------------------------------------------------------------------
app = Flask(__name__)

# Upload limits (16MB)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

OUTPUT_BASE_DIR = Path("embeddings")
INPUT_PPT_DIR = Path("input_ppt")
IMAGES_DIR = OUTPUT_BASE_DIR / "images"
MODEL_NAME = "dinov2"

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
INPUT_PPT_DIR.mkdir(parents=True, exist_ok=True)

# Dossier pour le stockage temporaire sur le serveur
TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# OpenAI client
client = OpenAI()

# --------------------------------------------------------------------------------------
# ENGINE (placeholder: adapte à ton code)
# --------------------------------------------------------------------------------------
ENGINE = None  # sera chargé via load_engine()

def load_engine(config_path: str):
    """
    Doit initialiser ENGINE (ex: FrenchMicrographSearchEngine)
    Adapte ici selon ta classe réelle.
    """
    global ENGINE
    print(f"📄 Loading search engine: {config_path}")
    # from your_engine_module import FrenchMicrographSearchEngine
    # ENGINE = FrenchMicrographSearchEngine(config_path=config_path)
    raise NotImplementedError("load_engine() doit être adapté à ton moteur de recherche.")


# --------------------------------------------------------------------------------------
# TEMP UPLOAD VALIDATION
# --------------------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "pdf", "txt", "csv", "xlsx", "docx", "pptx", "md", "json"
}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def guess_extension_from_mime(mime_type: Optional[str]) -> Optional[str]:
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


# --------------------------------------------------------------------------------------
# BACKGROUND CLEANUP TASK
# --------------------------------------------------------------------------------------
def cleanup_old_files(interval: int = 1800, max_age_seconds: int = 2 * 3600):
    """
    Supprime les fichiers du dossier temp_uploads s'ils ont plus de max_age_seconds.
    Vérifie toutes les `interval` secondes.
    """
    while True:
        now = time.time()
        try:
            for f in TEMP_UPLOAD_DIR.iterdir():
                if not f.is_file():
                    continue
                try:
                    age = now - f.stat().st_mtime
                    if age > max_age_seconds:
                        f.unlink(missing_ok=True)
                        print(f"🧹 Deleted old temp file: {f.name}")
                except Exception as e:
                    print(f"⚠️ Erreur lors de la suppression de {f.name} : {e}")
        except Exception as e:
            print(f"⚠️ Cleanup scan error: {e}")

        time.sleep(interval)

cleanup_thread = Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()


# --------------------------------------------------------------------------------------
# ROOT / HEALTH
# --------------------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "micrograph-search-api", "status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# --------------------------------------------------------------------------------------
# IMAGE SERVING
# --------------------------------------------------------------------------------------
@app.route("/images/<path:filename>", methods=["GET"])
def serve_image(filename):
    """Serve images from the embeddings/images directory"""
    try:
        return send_from_directory(str(IMAGES_DIR), filename)
    except Exception:
        return jsonify({"error": "not_found"}), 404

@app.route("/temp_files/<path:filename>", methods=["GET"])
def serve_temp_file(filename):
    """Sert les fichiers temporaires stockés localement"""
    try:
        return send_from_directory(str(TEMP_UPLOAD_DIR), filename)
    except Exception:
        return jsonify({"error": "temp_file_not_found"}), 404


# --------------------------------------------------------------------------------------
# LOCAL TEMPORARY STORAGE
# --------------------------------------------------------------------------------------
@app.route("/upload_temp_image", methods=["POST"])
def upload_temp_image():
    """
    Stockage temporaire local:
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
        return jsonify({
            "success": False,
            "error": "missing_openaiFileIdRefs",
            "message": "Provide openaiFileIdRefs (list) or legacy file_id",
        }), 400

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
                    except Exception:
                        pass

                file_bytes = client.files.content(file_id).read()

            # Filename sanitization + extension handling
            filename_safe = secure_filename(original_name or "uploaded_file")
            if not filename_safe:
                filename_safe = "uploaded_file"

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

            uploaded_results.append({
                "original_name": original_name,
                "filename": unique_filename,
                "url": file_url,
                "expires_in": "2 hours",
            })

        except Exception as e:
            print(f"❌ Error processing {file_ref}: {e}")
            errors.append(f"{file_ref}: {str(e)}")

    if not uploaded_results and errors:
        return jsonify({"success": False, "message": "All uploads failed", "errors": errors}), 500

    return jsonify({
        "success": True,
        "message": f"Processed {len(uploaded_results)} files.",
        "files": uploaded_results,
        "errors": errors,
    }), 200


# --------------------------------------------------------------------------------------
# SEARCH
# --------------------------------------------------------------------------------------
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

    top_k = int(data.get("top_k", 5))
    temp_filename = data.get("temp_filename")
    file_id = data.get("file_id")

    img = None

    # 1) temp_filename branch
    if temp_filename:
        file_path = TEMP_UPLOAD_DIR / temp_filename
        if not file_path.exists():
            return jsonify({"error": "temp_file_not_found", "message": f"{temp_filename} not found"}), 404
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            return jsonify({"error": "invalid_image", "message": str(e)}), 400

    # 2) OpenAI file_id branch
    elif file_id:
        try:
            file_info = client.files.retrieve(file_id)

            # Vérifier le purpose (selon ton usage)
            if getattr(file_info, "purpose", None) not in ["assistants", "vision", "assistants_output"]:
                return jsonify({
                    "error": "invalid_file_purpose",
                    "message": (
                        f"File purpose is '{getattr(file_info, 'purpose', None)}'. "
                        "Must be 'assistants' or 'vision' (or 'assistants_output')."
                    ),
                }), 400

            file_content = client.files.content(file_id).read()
            img = Image.open(io.BytesIO(file_content)).convert("RGB")

        except Exception as e:
            error_msg = str(e)
            if "No such File object" in error_msg or "Could not find" in error_msg:
                return jsonify({
                    "error": "file_not_accessible",
                    "message": (
                        "The file_id cannot be accessed. This usually means: "
                        "(1) The file is a conversation attachment, not uploaded via Files API, or "
                        "(2) The file has expired. Please upload using Files API with purpose='assistants'."
                    ),
                }), 400
            return jsonify({"error": "openai_retrieval_failed", "message": error_msg}), 400

    else:
        return jsonify({"error": "missing_input", "message": "Provide either file_id or temp_filename"}), 400

    # Run search
    try:
        results = ENGINE.search_from_pil(img, top_k=top_k)
        return jsonify({"success": True, "results": results}), 200
    except Exception as e:
        return jsonify({"success": False, "error": "search_failed", "message": str(e)}), 500


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
