from __future__ import annotations

import argparse
import io
import os
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import Optional, List, Dict, Any

import numpy as np
import requests
import torch
from flask import Flask, jsonify, request, send_from_directory, url_for
from openai import OpenAI
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from pgvector import Vector


DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"

# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# 🔥 IMPORTANT: Trust Azure reverse proxy headers
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# -----------------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_BASE_DIR = BASE_DIR / "embeddings_v7"
IMAGES_DIR = OUTPUT_BASE_DIR / "images"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# OPENAI CLIENT
# -----------------------------------------------------------------------------
client = OpenAI()

# -----------------------------------------------------------------------------
# DINOv2 (lazy load)
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_MODEL_NAME = "facebook/dinov2-large"

DINO_MODEL: Optional[AutoModel] = None
DINO_PROCESSOR: Optional[AutoImageProcessor] = None


def ensure_dino_loaded():
    global DINO_MODEL, DINO_PROCESSOR
    if DINO_MODEL is not None and DINO_PROCESSOR is not None:
        return
    print(f"🔧 Loading DINOv2 on {DEVICE}...")
    DINO_MODEL = AutoModel.from_pretrained(DINO_MODEL_NAME).to(DEVICE).eval()
    DINO_PROCESSOR = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
    print("✅ DINOv2 loaded")


def compute_embedding_from_pil(image: Image.Image) -> np.ndarray:
    ensure_dino_loaded()
    image = image.convert("RGB")
    inputs = DINO_PROCESSOR(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = DINO_MODEL(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding.astype("float32")


# -----------------------------------------------------------------------------
# CLEANUP THREAD
# -----------------------------------------------------------------------------
def cleanup_old_files(interval: int = 1800, max_age_seconds: int = 2 * 3600):
    while True:
        now = time.time()
        for f in TEMP_UPLOAD_DIR.iterdir():
            if f.is_file() and (now - f.stat().st_mtime > max_age_seconds):
                try:
                    f.unlink(missing_ok=True)
                    print(f"🧹 Deleted old temp file: {f.name}")
                except Exception as e:
                    print(f"⚠️ Error deleting {f.name}: {e}")
        time.sleep(interval)


Thread(target=cleanup_old_files, daemon=True).start()

# -----------------------------------------------------------------------------
# DB
# -----------------------------------------------------------------------------
def get_db_conn():
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    return conn


def search_similar_in_db(query_embedding: np.ndarray, top_k: int = 5):
    query_vec = Vector(query_embedding.tolist())

    sql = """
        SELECT
            mi.id,
            mi.image_path,
            mi.matiere_id,
            m.nom_matiere,
            m.reference,
            (1 - (mi.embedding <=> %s)) AS similarity
        FROM public.matiere_images mi
        JOIN public.matieres m ON m.matiere_id = mi.matiere_id
        ORDER BY mi.embedding <=> %s
        LIMIT %s;
    """

    conn = get_db_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (query_vec, query_vec, top_k))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# FILE SERVING
# -----------------------------------------------------------------------------
@app.route("/images/<path:filename>", methods=["GET"])
def serve_image(filename):
    return send_from_directory(str(IMAGES_DIR), filename)


@app.route("/temp_files/<path:filename>", methods=["GET"])
def serve_temp_file(filename):
    return send_from_directory(str(TEMP_UPLOAD_DIR), filename)


def build_image_url(filename: str) -> str:
    return url_for("serve_image", filename=secure_filename(filename), _external=True, _scheme="https")


# -----------------------------------------------------------------------------
# UPLOAD TEMP IMAGE
# -----------------------------------------------------------------------------
@app.route("/upload_temp_image", methods=["POST"])
def upload_temp_image():
    """Upload image from OpenAI Files API or download_link."""
    data = request.get_json(silent=True) or {}
    refs = data.get("openaiFileIdRefs")

    if not refs or not isinstance(refs, list):
        return jsonify({"success": False, "error": "missing_openaiFileIdRefs"}), 400

    uploaded_results = []
    errors = []

    for file_ref in refs:
        try:
            file_id = file_ref.get("id")
            download_link = file_ref.get("download_link")
            original_name = file_ref.get("name") or "uploaded_file"
            mime_type = file_ref.get("mime_type")

            if not file_id and not download_link:
                errors.append("Missing both 'id' and 'download_link'")
                continue

            file_bytes = None

            # Try to get file from OpenAI Files API
            if file_id:
                try:
                    file_bytes = client.files.content(file_id).read()
                    print(f"✅ Retrieved file {file_id} from OpenAI API")
                except Exception as openai_err:
                    print(f"⚠️ Failed to get file {file_id} from OpenAI: {openai_err}")
                    errors.append(f"OpenAI API error: {str(openai_err)}")
                    continue

            # Fallback to download_link if provided and we don't have file_bytes
            if not file_bytes and download_link:
                try:
                    response = requests.get(download_link, timeout=30)
                    response.raise_for_status()
                    file_bytes = response.content
                    print(f"✅ Downloaded file from link: {download_link}")
                except Exception as dl_err:
                    print(f"⚠️ Failed to download from link {download_link}: {dl_err}")
                    errors.append(f"Download link error: {str(dl_err)}")
                    continue

            if not file_bytes:
                errors.append(f"Could not retrieve file data for {original_name}")
                continue

            filename_safe = secure_filename(original_name)
            if "." not in filename_safe:
                filename_safe += ".png"

            unique_filename = f"{uuid.uuid4().hex}_{int(time.time())}_{filename_safe}"
            file_path = TEMP_UPLOAD_DIR / unique_filename

            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # ✅ Generate HTTPS URL for temporary file
            try:
                # Use url_for in request context for correct host resolution
                file_url = url_for(
                    "serve_temp_file",
                    filename=unique_filename,  # Don't call secure_filename again - already safe
                    _external=True,
                    _scheme="https"
                )
            except Exception as url_err:
                # Fallback URL construction if url_for fails
                print(f"⚠️ url_for failed: {url_err}, using manual URL construction")
                host = request.host if request else "micrographie-ia.azurewebsites.net"
                file_url = f"https://{host}/temp_files/{unique_filename}"
            
            print(f"✅ Generated URL: {file_url}")

            uploaded_results.append(
                {
                    "original_name": original_name,
                    "filename": unique_filename,
                    "url": file_url,
                    "expires_in": "2 hours",
                }
            )

        except Exception as e:
            print(f"❌ Unexpected error processing file ref: {e}")
            errors.append(f"Unexpected error: {str(e)}")

    # Return success if at least one file was processed
    success = len(uploaded_results) > 0
    status_code = 200  # Always return 200 per API spec
    
    return jsonify(
        {
            "success": success,
            "message": f"Processed {len(uploaded_results)} files.{f' {len(errors)} error(s).' if errors else ''}",
            "files": uploaded_results,
            "errors": errors,
        }
    ), status_code


# -----------------------------------------------------------------------------
# SEARCH
# -----------------------------------------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(silent=True) or {}
    download_link = data.get("download_link")
    top_k = int(data.get("top_k", 5))

    if not download_link:
        return jsonify({"success": False, "error": "missing_download_link"}), 400

    if not download_link.startswith("https://"):
        return jsonify({"success": False, "error": "download_link_must_be_https"}), 400

    try:
        r = requests.get(download_link, timeout=20)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "error": "download_link_failed", "message": str(e)}), 400

    query_embedding = compute_embedding_from_pil(img)
    rows = search_similar_in_db(query_embedding, top_k)

    results = []
    for r in rows:
        results.append(
            {
                "id": r["id"],
                "image_url": build_image_url(Path(r["image_path"]).name),
                "matiere_id": r["matiere_id"],
                "material_name": r["nom_matiere"],
                "reference": r["reference"],
                "similarity": float(r["similarity"]),
            }
        )

    return jsonify({"success": True, "results": results}), 200


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
