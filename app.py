from __future__ import annotations

import argparse
import io
import json
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


# =============================================================================
# CONFIG
# =============================================================================

DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"


# =============================================================================
# APP
# =============================================================================

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_BASE_DIR = BASE_DIR / "embeddings_v7"
IMAGES_DIR = OUTPUT_BASE_DIR / "images"
TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# OPENAI CLIENT
# =============================================================================

client = OpenAI()


# =============================================================================
# DINOv2 (lazy load)
# =============================================================================

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


# =============================================================================
# TEMP UPLOAD VALIDATION
# =============================================================================

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


# =============================================================================
# BACKGROUND CLEANUP TASK
# =============================================================================

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


# =============================================================================
# DB HELPERS
# =============================================================================

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


# =============================================================================
# ROOT / HEALTH
# =============================================================================

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


# =============================================================================
# FILE SERVING
# =============================================================================

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


# =============================================================================
# UPLOAD AND SEARCH (MERGED)
# =============================================================================

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
        return jsonify(
            {"success": False, "error": "invalid_top_k", "message": "top_k must be 1..50"}
        ), 400

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
                ext = guess_extension_from_mime(mime_type) or ".jpg"
                filename_safe += ext

            unique_name = f"{uuid.uuid4().hex}_{filename_safe}"
            save_path = TEMP_UPLOAD_DIR / unique_name

            with open(save_path, "wb") as f:
                f.write(file_bytes)

            # 3) Search
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

            # 4) Build local URL
            local_url = f"{request.host_url.rstrip('/')}/temp_files/{unique_name}"
            if local_url.startswith("http://"):
                local_url = "https://" + local_url[len("http://"):]

            final_results.append(
                {
                    "file_id": file_id,
                    "original_name": original_name,
                    "temp_filename": unique_name,
                    "temp_file_url": local_url,
                    "search_results": search_results,
                }
            )

        except Exception as e:
            errors.append(f"Error processing {file_ref.get('id', 'unknown')}: {str(e)}")

    return jsonify({"success": True, "results": final_results, "errors": errors}), 200


# =============================================================================
# APPLICATIONS ANALYSIS (NEW)
# =============================================================================

@app.route("/save_applications_analysis/<int:matiere_id>", methods=["POST"])
def save_applications_analysis(matiere_id):
    """
    Enregistre l'analyse JSON des applications générée par l'IA.
    """
    conn = None
    try:
        data = request.get_json(silent=True) or {}
        analysis_data = data.get("analysis_data")
        fiche_adn_id = data.get("fiche_adn_id")
        
        if not analysis_data:
            return jsonify({
                "success": False,
                "error": "missing_analysis_data",
                "message": "analysis_data (JSON) est requis dans le corps du POST"
            }), 400
        
        # Validation de la structure JSON
        if not isinstance(analysis_data, dict):
            return jsonify({
                "success": False,
                "error": "invalid_json",
                "message": "analysis_data doit être un objet JSON"
            }), 400
        
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Récupération des informations du matériau
            cur.execute("""
                SELECT matiere_id, nom_matiere, reference, type_matiere
                FROM public.matieres
                WHERE matiere_id = %s
            """, (matiere_id,))
            material = cur.fetchone()
            
            if not material:
                return jsonify({
                    "success": False,
                    "error": "material_not_found",
                    "message": f"Matériau {matiere_id} non trouvé"
                }), 404
            
            # Si fiche_adn_id n'est pas fourni, récupérer la plus récente
            if not fiche_adn_id:
                cur.execute("""
                    SELECT fiche_adn_id
                    FROM public.fiches_adn_matieres
                    WHERE matiere_id = %s
                    ORDER BY date_creation DESC
                    LIMIT 1
                """, (matiere_id,))
                fiche_result = cur.fetchone()
                if fiche_result:
                    fiche_adn_id = fiche_result["fiche_adn_id"]
            
            # Comptage des applications dans analysis_data
            num_apps = len(analysis_data.get("applications", []))
            
            # Insertion ou mise à jour (UPSERT)
            cur.execute("""
                INSERT INTO public.fiches_applications_matieres
                (matiere_id, fiche_adn_id, nom_matiere, reference, type_matiere,
                 analysis_data, num_applications, date_creation, derniere_modification)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (matiere_id)
                DO UPDATE SET
                    analysis_data = EXCLUDED.analysis_data,
                    num_applications = EXCLUDED.num_applications,
                    fiche_adn_id = COALESCE(EXCLUDED.fiche_adn_id, fiches_applications_matieres.fiche_adn_id),
                    derniere_modification = CURRENT_TIMESTAMP
                RETURNING fiche_app_id;
            """, (
                matiere_id,
                fiche_adn_id,
                material["nom_matiere"],
                material["reference"],
                material["type_matiere"],
                json.dumps(analysis_data, ensure_ascii=False),
                num_apps
            ))
            
            result = cur.fetchone()
            fiche_app_id = result["fiche_app_id"] if result else None
            
            conn.commit()
            
            return jsonify({
                "success": True,
                "message": "Analyse des applications enregistrée avec succès",
                "fiche_app_id": fiche_app_id,
                "matiere_id": matiere_id,
                "num_applications": num_apps
            }), 201
    
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ Erreur: {e}")
        return jsonify({"success": False, "error": "save_failed", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()


@app.route("/get_applications_analysis/<int:matiere_id>", methods=["GET"])
def get_applications_analysis(matiere_id):
    """
    Récupère l'analyse des applications enregistrée pour un matériau.
    """
    conn = None
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    fiche_app_id, matiere_id, fiche_adn_id, nom_matiere,
                    reference, type_matiere, analysis_data, num_applications,
                    date_creation, derniere_modification
                FROM public.fiches_applications_matieres
                WHERE matiere_id = %s
            """, (matiere_id,))
            
            result = cur.fetchone()
            
            if not result:
                return jsonify({
                    "success": False,
                    "error": "analysis_not_found",
                    "message": f"Aucune analyse trouvée pour matiere_id {matiere_id}"
                }), 404
            
            return jsonify({
                "success": True,
                "fiche_app": {
                    "fiche_app_id": result["fiche_app_id"],
                    "matiere_id": result["matiere_id"],
                    "fiche_adn_id": result["fiche_adn_id"],
                    "nom_matiere": result["nom_matiere"],
                    "reference": result["reference"],
                    "type_matiere": result["type_matiere"],
                    "num_applications": result["num_applications"],
                    "date_creation": result["date_creation"],
                    "derniere_modification": result["derniere_modification"],
                    "analysis_data": result["analysis_data"]
                }
            }), 200
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return jsonify({"success": False, "error": "retrieval_failed", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()


# =============================================================================
# FICHE ADN - GET BY REFERENCE
# =============================================================================

@app.route("/fiche_adn/reference/<string:reference>", methods=["GET"])
def get_fiche_adn_by_reference(reference):
    """
    Get the complete ADN specifications sheet for a material by its reference.
    """
    conn = None
    try:
        conn = get_db_conn()

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    fiche_adn_id,
                    matiere_id,
                    nom_matiere,
                    reference,
                    type_matiere,
                    specifications,
                    num_specifications,
                    date_creation,
                    derniere_modification
                FROM public.fiches_ADN_matieres
                WHERE UPPER(REPLACE(TRIM(reference), ' ', '')) = UPPER(REPLACE(%s, ' ', ''))
                LIMIT 1
                """,
                (reference,),
            )

            result = cur.fetchone()

            if not result:
                return jsonify(
                    {
                        "success": False,
                        "error": "fiche_adn_not_found",
                        "message": f"No fiche ADN found for reference: {reference}",
                    }
                ), 404

            result_dict = dict(result)

            return jsonify(
                {
                    "success": True,
                    "fiche_adn": {
                        "fiche_adn_id": result_dict["fiche_adn_id"],
                        "matiere_id": result_dict["matiere_id"],
                        "nom_matiere": result_dict["nom_matiere"],
                        "reference": result_dict["reference"],
                        "type_matiere": result_dict["type_matiere"],
                        "num_specifications": result_dict["num_specifications"],
                        "date_creation": result_dict["date_creation"],
                        "derniere_modification": result_dict["derniere_modification"],
                        "specifications": result_dict["specifications"],
                    },
                }
            ), 200

    except Exception as e:
        return jsonify({"success": False, "error": "retrieval_failed", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()


# =============================================================================
# FICHE ADN - GET BY ID
# =============================================================================

@app.route("/fiche_adn/<int:matiere_id>", methods=["GET"])
def get_fiche_adn_by_id(matiere_id):
    """
    Get the complete ADN specifications sheet for a material by matiere_id.
    """
    conn = None
    try:
        conn = get_db_conn()

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    fiche_adn_id,
                    matiere_id,
                    nom_matiere,
                    reference,
                    type_matiere,
                    specifications,
                    num_specifications,
                    date_creation,
                    derniere_modification
                FROM public.fiches_ADN_matieres
                WHERE matiere_id = %s
                LIMIT 1
                """,
                (matiere_id,),
            )

            result = cur.fetchone()

            if not result:
                return jsonify(
                    {
                        "success": False,
                        "error": "fiche_adn_not_found",
                        "message": f"No fiche ADN found for matiere_id: {matiere_id}",
                    }
                ), 404

            result_dict = dict(result)

            return jsonify(
                {
                    "success": True,
                    "fiche_adn": {
                        "fiche_adn_id": result_dict["fiche_adn_id"],
                        "matiere_id": result_dict["matiere_id"],
                        "nom_matiere": result_dict["nom_matiere"],
                        "reference": result_dict["reference"],
                        "type_matiere": result_dict["type_matiere"],
                        "num_specifications": result_dict["num_specifications"],
                        "date_creation": result_dict["date_creation"],
                        "derniere_modification": result_dict["derniere_modification"],
                        "specifications": result_dict["specifications"],
                    },
                }
            ), 200

    except Exception as e:
        return jsonify({"success": False, "error": "retrieval_failed", "message": str(e)}), 500
    finally:
        if conn:
            conn.close()


# =============================================================================
# SEARCH (BACKWARD COMPATIBILITY)
# =============================================================================

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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=args.debug)
