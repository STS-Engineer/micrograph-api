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
    """
    Upload image from OpenAI Files API and generate HTTPS download link.
    
    Request: {"openaiFileIdRefs": [{"id": "file-xxx", "name": "image.png"}]}
    """
    data = request.get_json(silent=True) or {}
    refs = data.get("openaiFileIdRefs")

    if not refs or not isinstance(refs, list):
        return jsonify({"success": False, "error": "missing_openaiFileIdRefs"}), 400

    uploaded_results = []
    errors = []

    for file_ref in refs:
        try:
            file_id = file_ref.get("id")
            original_name = file_ref.get("name") or "uploaded_file"

            if not file_id:
                errors.append("Missing 'id' field")
                continue

            # Retrieve file from OpenAI Files API
            try:
                print(f"📥 Retrieving file {file_id} from OpenAI...")
                file_content = client.files.content(file_id)
                file_bytes = file_content.read()
                print(f"✅ Retrieved file {file_id} from OpenAI API ({len(file_bytes)} bytes)")
            except Exception as openai_err:
                print(f"❌ OpenAI error for {file_id}: {openai_err}")
                errors.append(f"OpenAI error: {str(openai_err)}")
                continue

            # Save to temporary file
            filename_safe = secure_filename(original_name)
            if "." not in filename_safe:
                filename_safe += ".png"

            unique_filename = f"{uuid.uuid4().hex}_{int(time.time())}_{filename_safe}"
            file_path = TEMP_UPLOAD_DIR / unique_filename

            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # Generate HTTPS download link
            scheme = request.headers.get("X-Forwarded-Proto", "https")
            host = request.headers.get("X-Forwarded-Host") or request.host
            file_url = f"{scheme}://{host}/temp_files/{unique_filename}"
            
            print(f"✅ Generated HTTPS URL: {file_url}")

            uploaded_results.append({
                "original_name": original_name,
                "filename": unique_filename,
                "url": file_url,
                "expires_in": "2 hours",
            })

        except Exception as e:
            print(f"❌ Unexpected error processing file: {e}")
            errors.append(f"Unexpected error: {str(e)}")

    # Return response
    success = len(uploaded_results) > 0
    
    return jsonify({
        "success": success,
        "message": f"Processed {len(uploaded_results)} files.{f' {len(errors)} error(s).' if errors else ''}",
        "files": uploaded_results,
        "errors": errors,
    }), 200


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
