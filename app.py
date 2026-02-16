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
from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from werkzeug.utils import secure_filename

import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from pgvector import Vector

DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"


# -----------------------------------------------------------------------------
# APP
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB


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
    """Compute DINOv2 embedding (1024 dims)."""
    ensure_dino_loaded()

    image = image.convert("RGB")
    inputs = DINO_PROCESSOR(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = DINO_MODEL(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    return embedding.astype("float32")


# -----------------------------------------------------------------------------
# TEMP UPLOAD VALIDATION
# -----------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


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
# DB HELPERS
# -----------------------------------------------------------------------------
def get_db_conn():
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    return conn


def search_similar_in_db(query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Returns top_k similar images from pgvector.
    Uses cosine distance (<=>).
    similarity = 1 - distance
    """
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
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()


def build_image_url(image_path: str) -> str:
    """
    image_path stored in DB is like: embeddings_v7/images/xxx.png
    We want to expose it through /images/<filename>.
    """
    filename = Path(image_path).name
    url = f"{request.host_url.rstrip('/')}/images/{secure_filename(filename)}"
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


# -----------------------------------------------------------------------------
# ROOT / HEALTH
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "micrograph-search-api",
            "status": "ok",
            "model": DINO_MODEL_NAME,
            "dino_loaded": DINO_MODEL is not None,
            "images_dir": str(IMAGES_DIR),
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "dino_loaded": DINO_MODEL is not None}), 200


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
# UPLOAD AND SEARCH (MERGED)
# -----------------------------------------------------------------------------
@app.route("/upload_and_search", methods=["POST"])
def upload_and_search():
    """
    Merged endpoint:
    1) Receives openaiFileIdRefs (list of file references).
    2) Downloads and saves them to temp_uploads/.
    3) For each file, computes embedding and searches for similar images.
    4) Returns both the local file info and the search results.
    """
    data = request.get_json(silent=True) or {}
    refs = data.get("openaiFileIdRefs")
    top_k = int(data.get("top_k", 5))

    if not refs or not isinstance(refs, list):
        return jsonify(
            {
                "success": False,
                "error": "missing_openaiFileIdRefs",
                "message": "Provide openaiFileIdRefs (list).",
            }
        ), 400

    if top_k < 1 or top_k > 50:
        return jsonify({"success": False, "error": "invalid_top_k", "message": "top_k must be 1..50"}), 400

    final_results = []
    errors = []

    for file_ref in refs:
        try:
            if not isinstance(file_ref, dict):
                errors.append("Each item in openaiFileIdRefs must be an object.")
                continue

            file_id = file_ref.get("id")
            download_link = file_ref.get("download_link")
            original_name = file_ref.get("name") or "uploaded_file"
            mime_type = file_ref.get("mime_type")

            if not file_id:
                errors.append("Missing id in file reference.")
                continue

            file_bytes = None

            # 1) Try direct link if provided
            if download_link:
                try:
                    r = requests.get(download_link, timeout=20)
                    r.raise_for_status()
                    file_bytes = r.content
                except Exception as e:
                    print(f"⚠️ download_link failed, fallback to file_id: {e}")

            # 2) Fallback: OpenAI file content
            if file_bytes is None:
                file_bytes = client.files.content(file_id).read()

            filename_safe = secure_filename(original_name or "uploaded_file") or "uploaded_file"

            if "." not in filename_safe:
                ext = guess_extension_from_mime(mime_type) or ".png"
                filename_safe += ext

            if not allowed_file(filename_safe):
                errors.append(f"{original_name}: File type not allowed (png/jpg/jpeg only).")
                continue

            # Save to temp
            unique_filename = f"{uuid.uuid4().hex}_{int(time.time())}_{filename_safe}"
            file_path = TEMP_UPLOAD_DIR / unique_filename
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # Generate local URL
            file_url = f"{request.host_url.rstrip('/')}/temp_files/{unique_filename}"
            if file_url.startswith("http://"):
                file_url = "https://" + file_url[len("http://"):]

            # --- SEARCH PART ---
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            query_embedding = compute_embedding_from_pil(img)
            rows = search_similar_in_db(query_embedding, top_k=top_k)

            search_results = []
            for r in rows:
                search_results.append(
                    {
                        "id": r["id"],
                        "image_url": build_image_url(r["image_path"]),
                        "matiere_id": r["matiere_id"],
                        "material_name": r["nom_matiere"],
                        "reference": r["reference"],
                        "similarity": float(r["similarity"]) if r["similarity"] is not None else None,
                    }
                )

            final_results.append(
                {
                    "original_name": original_name,
                    "filename": unique_filename,
                    "url": file_url,
                    "expires_in": "2 hours",
                    "search_results": search_results
                }
            )

        except Exception as e:
            print(f"❌ Error processing file_ref: {e}")
            errors.append(f"{file_ref}: {str(e)}")

    if not final_results and errors:
        return jsonify({"success": False, "message": "All operations failed", "errors": errors}), 500

    return jsonify(
        {
            "success": True,
            "message": f"Processed {len(final_results)} files with search results.",
            "results": final_results,
            "errors": errors,
        }
    ), 200


# -----------------------------------------------------------------------------
# MATERIAL DETAILS
# -----------------------------------------------------------------------------
@app.route("/material_details/<int:matiere_id>", methods=["GET"])
def get_material_details(matiere_id):
    """
    Get complete material information by matiere_id.
    Returns: matieres + fiches_matieres + specifications + expert_notes
    """
    conn = None
    try:
        conn = get_db_conn()

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM public.matieres WHERE matiere_id = %s", (matiere_id,))
            material = cur.fetchone()

            if not material:
                return jsonify(
                    {"success": False, "error": "material_not_found", "message": f"matiere_id {matiere_id} not found"}
                ), 404

            material = dict(material)

            cur.execute(
                """
                SELECT fiche_id, date_creation_fiche, derniere_modification
                FROM public.fiches_matieres
                WHERE matiere_id = %s
                ORDER BY fiche_id DESC
                """,
                (matiere_id,),
            )
            fiches = [dict(row) for row in cur.fetchall()]

            specifications = []
            for fiche in fiches:
                cur.execute(
                    """
                    SELECT spec_id, fiche_id, source_type, donnees, date_creation, derniere_modification
                    FROM public.specifications
                    WHERE fiche_id = %s
                    ORDER BY spec_id
                    """,
                    (fiche["fiche_id"],),
                )
                specifications.extend([dict(row) for row in cur.fetchall()])

            cur.execute(
                """
                SELECT men.id, men.matiere_image_id, men.note_json, men.created_at
                FROM public.matiere_expert_notes men
                INNER JOIN public.matiere_images mi ON mi.id = men.matiere_image_id
                WHERE mi.matiere_id = %s
                ORDER BY men.created_at DESC
                """,
                (matiere_id,),
            )
            expert_notes = [dict(row) for row in cur.fetchall()]

            response = {
                "success": True,
                "material": material,
                "fiches_matieres": fiches,
                "specifications": specifications,
                "expert_notes": expert_notes,
                "summary": {
                    "matiere_id": matiere_id,
                    "nom_matiere": material.get("nom_matiere"),
                    "reference": material.get("reference"),
                    "type_matiere": material.get("type_matiere"),
                    "num_fiches": len(fiches),
                    "num_specifications": len(specifications),
                    "num_expert_notes": len(expert_notes),
                },
            }

            return jsonify(response), 200

    except Exception as e:
        return jsonify({"success": False, "error": "retrieval_failed", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()


# -----------------------------------------------------------------------------
# SEARCH (KEEPING FOR BACKWARD COMPATIBILITY)
# -----------------------------------------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    """
    Search similar micrographs (pgvector).
    Accepts ONE of:
      - download_link (recommended: robust for autoscaling)
      - temp_filename (fallback: local file)
      - file_id (fallback: OpenAI Files API)
    """
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"success": False, "error": "missing_json_body", "message": "Missing JSON body"}), 400

    top_k = int(data.get("top_k", 5))
    if top_k < 1 or top_k > 50:
        return jsonify({"success": False, "error": "invalid_top_k", "message": "top_k must be 1..50"}), 400

    temp_filename = data.get("temp_filename")
    file_id = data.get("file_id")
    download_link = data.get("download_link")

    provided = [bool(download_link), bool(temp_filename), bool(file_id)]
    if sum(provided) != 1:
        return jsonify(
            {
                "success": False,
                "error": "invalid_input",
                "message": "Provide exactly ONE of: download_link, temp_filename, file_id",
            }
        ), 400

    img = None

    # 1) download_link (BEST)
    if download_link:
        try:
            r = requests.get(download_link, timeout=20)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            return jsonify({"success": False, "error": "download_link_failed", "message": str(e)}), 400

    # 2) temp_filename (fallback)
    elif temp_filename:
        file_path = TEMP_UPLOAD_DIR / temp_filename
        if not file_path.exists():
            return jsonify(
                {
                    "success": False,
                    "error": "temp_file_not_found",
                    "message": f"{temp_filename} not found (likely autoscaling issue). Use download_link instead.",
                }
            ), 404
        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            return jsonify({"success": False, "error": "invalid_image", "message": str(e)}), 400

    # 3) file_id (fallback)
    elif file_id:
        try:
            file_content = client.files.content(file_id).read()
            img = Image.open(io.BytesIO(file_content)).convert("RGB")
        except Exception as e:
            return jsonify({"success": False, "error": "openai_retrieval_failed", "message": str(e)}), 400

    try:
        query_embedding = compute_embedding_from_pil(img)
        rows = search_similar_in_db(query_embedding, top_k=top_k)

        results = []
        for r in rows:
            results.append(
                {
                    "id": r["id"],
                    "image_url": build_image_url(r["image_path"]),
                    "matiere_id": r["matiere_id"],
                    "material_name": r["nom_matiere"],
                    "reference": r["reference"],
                    "similarity": float(r["similarity"]) if r["similarity"] is not None else None,
                }
            )

        return jsonify({"success": True, "results": results}), 200

    except Exception as e:
        return jsonify({"success": False, "error": "search_failed", "message": str(e)}), 500


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
