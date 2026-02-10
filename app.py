from __future__ import annotations

import argparse
import io
import os
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

# ✅ Your engine (as provided)
from search_similar_french_v2 import FrenchMicrographSearchEngine  # :contentReference[oaicite:3]{index=3}

# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# -----------------------------------------------------------------------------
# PATHS (robust on Azure/Gunicorn)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ✅ Matches your repo folder name
OUTPUT_BASE_DIR = BASE_DIR / "embeddings_v7"
IMAGES_DIR = OUTPUT_BASE_DIR / "images"
MODEL_NAME = "dinov2"

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# OPENAI CLIENT (OPENAI_API_KEY via env)
# -----------------------------------------------------------------------------
client = OpenAI()

# -----------------------------------------------------------------------------
# ENGINE (lazy load per worker)
# -----------------------------------------------------------------------------
ENGINE: Optional[FrenchMicrographSearchEngine] = None


def load_engine(config_path: str) -> FrenchMicrographSearchEngine:
    """Instantiate the search engine from config path."""
    global ENGINE
    print(f"📄 Loading search engine: {config_path}")
    ENGINE = FrenchMicrographSearchEngine(config_path=config_path)
    return ENGINE


def ensure_engine_loaded() -> None:
    """Lazy-load engine (each Gunicorn worker loads it when needed)."""
    global ENGINE
    if ENGINE is not None:
        return

    config_path = OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing engine config: {config_path}. "
            "Make sure embeddings_v7/ is deployed with search_config_dinov2.json."
        )

    load_engine(str(config_path))
    print("✅ ENGINE loaded")


# Optional autoload (won't crash app if it fails)
try:
    print("CWD:", Path().resolve())
    print("BASE_DIR:", BASE_DIR)
    print("OUTPUT_BASE_DIR:", OUTPUT_BASE_DIR)
    print("CONFIG:", OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
    print("CONFIG EXISTS:", (OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json").exists())
    ensure_engine_loaded()
except Exception as e:
    print(f"⚠️ Engine auto-load failed on import: {e}")
    ENGINE = None

# -----------------------------------------------------------------------------
# TEMP UPLOAD VALIDATION
# -----------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "pdf", "txt", "csv", "xlsx", "docx", "pptx", "md", "json"}


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


# -----------------------------------------------------------------------------
# BACKGROUND CLEANUP TASK
# -----------------------------------------------------------------------------
def cleanup_old_files(interval: int = 1800, max_age_seconds: int = 2 * 3600):
    """Deletes files in temp_uploads/ older than max_age_seconds."""
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
                    print(f"⚠️ Error deleting {f.name}: {e}")
        except Exception as e:
            print(f"⚠️ Cleanup scan error: {e}")

        time.sleep(interval)


cleanup_thread = Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()

# -----------------------------------------------------------------------------
# ROOT / HEALTH
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "micrograph-search-api",
            "status": "ok",
            "model": MODEL_NAME,
            "engine_loaded": ENGINE is not None,
            "output_dir": str(OUTPUT_BASE_DIR),
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "engine_loaded": ENGINE is not None}), 200


# -----------------------------------------------------------------------------
# FILE SERVING
# -----------------------------------------------------------------------------
@app.route("/images/<path:filename>", methods=["GET"])
def serve_image(filename):
    """Serve images from embeddings_v7/images"""
    try:
        return send_from_directory(str(IMAGES_DIR), filename)
    except Exception:
        return jsonify({"error": "not_found"}), 404


@app.route("/temp_files/<path:filename>", methods=["GET"])
def serve_temp_file(filename):
    """Serve locally stored temp uploads"""
    try:
        return send_from_directory(str(TEMP_UPLOAD_DIR), filename)
    except Exception:
        return jsonify({"error": "temp_file_not_found"}), 404


# -----------------------------------------------------------------------------
# LOCAL TEMP UPLOAD (OpenAI file_id -> local temp file)
# -----------------------------------------------------------------------------
@app.route("/upload_temp_image", methods=["POST"])
def upload_temp_image():
    """
    Receives:
      - openaiFileIdRefs: [ {id, download_link?, name?, mime_type?}, ... ] or ["file-..."]
      - (compat) {"file_id": "..."}
    Saves into temp_uploads/ and returns local URLs /temp_files/<filename>
    """
    data = request.get_json(silent=True) or {}

    refs = data.get("openaiFileIdRefs")
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

            file_bytes = None

            # 1) try direct link
            if download_link:
                try:
                    r = requests.get(download_link, timeout=15)
                    r.raise_for_status()
                    file_bytes = r.content
                except Exception as e:
                    print(f"⚠️ download_link failed, falling back to file_id: {e}")

            # 2) fallback OpenAI file content
            if file_bytes is None:
                # try retrieve filename if missing
                if not original_name or original_name == "uploaded_file":
                    try:
                        file_info = client.files.retrieve(file_id)
                        if getattr(file_info, "filename", None):
                            original_name = file_info.filename
                    except Exception:
                        pass

                file_bytes = client.files.content(file_id).read()

            filename_safe = secure_filename(original_name or "uploaded_file") or "uploaded_file"

            if "." not in filename_safe:
                ext = guess_extension_from_mime(mime_type) or ".bin"
                filename_safe += ext

            if not allowed_file(filename_safe):
                errors.append(f"{original_name}: File type not allowed")
                continue

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


# -----------------------------------------------------------------------------
# SEARCH
# -----------------------------------------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    """
    Search similar micrographs.
    Accepts either:
      - temp_filename (from /upload_temp_image)
      - file_id (OpenAI Files API)
    """
    try:
        ensure_engine_loaded()
    except Exception as e:
        return jsonify({"error": "engine_not_loaded", "message": str(e)}), 500

    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"error": "missing_json_body"}), 400

    top_k = int(data.get("top_k", 5))
    temp_filename = data.get("temp_filename")
    file_id = data.get("file_id")

    img = None

    if temp_filename:
        file_path = TEMP_UPLOAD_DIR / temp_filename
        if not file_path.exists():
            return jsonify({"error": "temp_file_not_found", "message": f"{temp_filename} not found"}), 404
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            return jsonify({"error": "invalid_image", "message": str(e)}), 400

    elif file_id:
        try:
            file_info = client.files.retrieve(file_id)
            if getattr(file_info, "purpose", None) not in ["assistants", "vision", "assistants_output"]:
                return jsonify(
                    {
                        "error": "invalid_file_purpose",
                        "message": (
                            f"File purpose is '{getattr(file_info, 'purpose', None)}'. "
                            "Must be 'assistants' or 'vision' (or 'assistants_output')."
                        ),
                    }
                ), 400

            file_content = client.files.content(file_id).read()
            img = Image.open(io.BytesIO(file_content)).convert("RGB")

        except Exception as e:
            error_msg = str(e)
            if "No such File object" in error_msg or "Could not find" in error_msg:
                return jsonify(
                    {
                        "error": "file_not_accessible",
                        "message": (
                            "The file_id cannot be accessed. This usually means: "
                            "(1) the file is a conversation attachment (not Files API), or "
                            "(2) it expired. Upload via Files API with purpose='assistants'."
                        ),
                    }
                ), 400
            return jsonify({"error": "openai_retrieval_failed", "message": error_msg}), 400

    else:
        return jsonify({"error": "missing_input", "message": "Provide either file_id or temp_filename"}), 400

    try:
        results = ENGINE.search_from_pil(img, top_k=top_k)  # :contentReference[oaicite:4]{index=4}
        return jsonify({"success": True, "results": results}), 200
    except Exception as e:
        return jsonify({"success": False, "error": "search_failed", "message": str(e)}), 500


# -----------------------------------------------------------------------------
# OPTIONAL: reload engine endpoint
# -----------------------------------------------------------------------------
@app.route("/reload_engine", methods=["POST"])
def reload_engine():
    global ENGINE
    try:
        ENGINE = None
        ensure_engine_loaded()
        return jsonify({"success": True, "engine_loaded": True}), 200
    except Exception as e:
        ENGINE = None
        return jsonify({"success": False, "error": "engine_reload_failed", "message": str(e)}), 500


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
