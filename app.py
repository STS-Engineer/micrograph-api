import os
import argparse
import shutil
import json
import torch
import time
import uuid
import io
import gc
import logging
import psycopg2
from pathlib import Path
from threading import Thread
from typing import Optional

import requests
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from openai import OpenAI
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# -----------------------------
# IMPORTS LOCAUX
# -----------------------------
try:
    from extract_references_french_v3_3 import process_powerpoint
    from compute_embeddings import EmbeddingComputer, save_embeddings
    from build_faiss_index_proper import build_faiss_index
    from search_similar_french_v2 import FrenchMicrographSearchEngine
except ImportError as e:
    print(f"⚠️ Attention: Certains modules locaux sont manquants : {e}")
    # L'app peut démarrer, mais /update_index et /search ne fonctionneront pas sans ces modules.

# -----------------------------
# CONFIGURATION DU LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# -----------------------------
# CONFIGURATION GLOBALE
# -----------------------------
ENGINE = None

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA",
)

OUTPUT_BASE_DIR = Path("embeddings_v7")
IMAGES_DIR = OUTPUT_BASE_DIR / "images"
MODEL_NAME = "dinov2"

TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Upload limits (16MB)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# Initialisation du client OpenAI (API key via env OPENAI_API_KEY)
client = OpenAI()

# -----------------------------
# TEMP UPLOAD VALIDATION
# -----------------------------
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "pdf",
    "txt",
    "csv",
    "xlsx",
    "docx",
    "pptx",
    "md",
    "json",
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


# -----------------------------
# DATABASE
# -----------------------------
def get_ppt_files_from_db():
    """
    Récupère les chemins des PowerPoints depuis la table `powerpoint_files`
    (colonne `file_path`).
    IMPORTANT: les chemins doivent être accessibles depuis la machine serveur.
    """
    paths = []
    conn = None
    try:
        logger.info("📡 Connexion à Azure PostgreSQL...")
        conn = psycopg2.connect(DB_URL, sslmode="require")
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM powerpoint_files")
        rows = cur.fetchall()
        paths = [Path(row[0].replace('"', "").strip()) for row in rows if row and row[0]]
        cur.close()
        logger.info(f"✅ DB : {len(paths)} fichiers récupérés.")
    except Exception as e:
        logger.error(f"❌ Erreur DB : {e}")
    finally:
        if conn:
            conn.close()
    return paths


# -----------------------------
# ENGINE
# -----------------------------
def load_engine(config_path: str):
    global ENGINE
    try:
        logger.info(f"⚙️ Chargement du moteur : {config_path}")
        if os.path.exists(config_path):
            ENGINE = FrenchMicrographSearchEngine(config_path=config_path)
            logger.info("✅ Moteur de recherche prêt !")
        else:
            logger.warning(f"⚠️ Fichier de configuration introuvable : {config_path}")
            ENGINE = None
    except Exception as e:
        logger.error(f"⚠️ Échec du chargement du moteur : {e}")
        ENGINE = None


# Auto-load engine on import (useful for gunicorn)
existing_config = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
if os.path.exists(existing_config):
    try:
        load_engine(existing_config)
        logger.info("✅ Moteur auto-chargé au démarrage (import/gunicorn).")
    except Exception as e:
        logger.warning(f"⚠️ Auto-load moteur échoué : {e}")


# -----------------------------
# NETTOYAGE AUTO (BACKGROUND)
# -----------------------------
def cleanup_old_files(interval=1800):
    """
    Supprime les fichiers du dossier temp_uploads s'ils ont plus de 2 heures.
    """
    while True:
        try:
            now = time.time()
            for f in TEMP_UPLOAD_DIR.glob("*"):
                if f.is_file() and (now - f.stat().st_mtime > 7200):
                    try:
                        f.unlink()
                        logger.info(f"🗑️ Fichier temporaire supprimé : {f.name}")
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage : {e}")
        time.sleep(interval)


Thread(target=cleanup_old_files, daemon=True).start()

# -----------------------------
# ROUTES API
# -----------------------------


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "status": "ok",
            "service": "micrographie-ia",
            "engine_ready": ENGINE is not None,
            "model": MODEL_NAME,
            "endpoints": [
                "/health",
                "/search",
                "/update_index",
                "/upload_temp_image",
                "/temp_files/<filename>",
                "/uploads/<filename>",
            ],
        }
    ), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Route pour vérifier si l'API est en ligne et si le moteur est chargé."""
    return (
        jsonify(
            {
                "status": "ok",
                "engine_ready": ENGINE is not None,
                "model": MODEL_NAME,
                "message": "Micrograph API is running",
            }
        ),
        200,
    )


# -----------------------------
# IMAGE SERVING
# -----------------------------
@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_indexed_image(filename):
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
# LOCAL TEMPORARY STORAGE
# -----------------------------
@app.route("/upload_temp_image", methods=["POST"])
def upload_temp_image():
    """
    Stockage temporaire local (comme app.py 1), compatible avec plusieurs formats :
    - openaiFileIdRefs: [ {id, download_link?, name?, mime_type?}, ... ] ou ["file-..."]
    - compat: {"file_id": "..."}
    Télécharge bytes (download_link -> fallback OpenAI file_id),
    valide l'extension, sauvegarde dans temp_uploads/,
    retourne URL locale /temp_files/<filename>.
    """
    data = request.get_json(silent=True) or {}
    refs = data.get("openaiFileIdRefs")

    # Backward-compat
    if not refs and data.get("file_id"):
        refs = [{"id": data["file_id"], "name": None, "download_link": None, "mime_type": None}]

    if not refs or not isinstance(refs, list):
        return (
            jsonify(
                {
                    "success": False,
                    "error": "missing_openaiFileIdRefs",
                    "message": "Provide openaiFileIdRefs (list) or legacy file_id",
                }
            ),
            400,
        )

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

            file_bytes = None

            # Try download_link first (if any)
            if download_link:
                try:
                    logger.info(f"⬇️ Trying download_link for {original_name}")
                    r = requests.get(download_link, timeout=15)
                    r.raise_for_status()
                    file_bytes = r.content
                except Exception as e:
                    logger.warning(f"⚠️ download_link failed, falling back to file_id: {e}")

            # Fallback: OpenAI file content
            if file_bytes is None:
                # If name missing, try retrieve filename from OpenAI
                if not original_name or original_name == "uploaded_file":
                    try:
                        file_info = client.files.retrieve(file_id)
                        if getattr(file_info, "filename", None):
                            original_name = file_info.filename
                    except Exception:
                        pass

                file_bytes = client.files.content(file_id).read()

            # Sanitize filename + ensure extension
            filename_safe = secure_filename(original_name or "uploaded_file")
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
            logger.error(f"❌ Error processing {file_ref}: {e}")
            errors.append(f"{file_ref}: {str(e)}")

    if not uploaded_results and errors:
        return jsonify({"success": False, "message": "All uploads failed", "errors": errors}), 500

    return (
        jsonify(
            {
                "success": True,
                "message": f"Processed {len(uploaded_results)} files.",
                "files": uploaded_results,
                "errors": errors,
            }
        ),
        200,
    )


# -----------------------------
# SEARCH
# -----------------------------
@app.route("/search", methods=["POST"])
def search():
    if ENGINE is None:
        return jsonify({"error": "engine_not_ready"}), 503

    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"error": "missing_json_body"}), 400

    top_k = data.get("top_k", 3)

    try:
        img = None

        # Option A: temp_filename (uploaded via /upload_temp_image)
        if "temp_filename" in data and data["temp_filename"]:
            img_path = TEMP_UPLOAD_DIR / data["temp_filename"]
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    return jsonify({"error": "invalid_temp_file", "message": str(e)}), 400
            else:
                return jsonify({"error": "temp_file_expired_or_not_found"}), 404

        # Option B: OpenAI file_id
        elif "file_id" in data and data["file_id"]:
            file_id = data["file_id"]
            try:
                file_info = client.files.retrieve(file_id)

                # Validate purpose (same spirit as app.py 1)
                if getattr(file_info, "purpose", None) not in ["assistants", "vision", "assistants_output"]:
                    return (
                        jsonify(
                            {
                                "error": "invalid_file_purpose",
                                "message": f"File purpose is '{getattr(file_info, 'purpose', None)}'. Must be 'assistants' or 'vision'. Please re-upload with correct purpose.",
                            }
                        ),
                        400,
                    )

                file_content = client.files.content(file_id).read()
                img = Image.open(io.BytesIO(file_content)).convert("RGB")

            except Exception as e:
                error_msg = str(e)
                # Message clair quand le file_id n'est pas accessible
                if "No such File object" in error_msg or "Could not find" in error_msg:
                    return (
                        jsonify(
                            {
                                "error": "file_not_accessible",
                                "message": "The file_id cannot be accessed. It may be expired or not uploaded via OpenAI Files API. Please upload with purpose='assistants'.",
                            }
                        ),
                        400,
                    )
                return jsonify({"error": "openai_retrieval_failed", "message": error_msg}), 400

        else:
            return jsonify({"error": "no_input_provided", "message": "Provide temp_filename or file_id"}), 400

        results = ENGINE.search_from_pil(img, top_k=top_k)
        return jsonify({"results": results}), 200

    except Exception as e:
        logger.error(f"Erreur lors de la recherche : {e}", exc_info=True)
        return jsonify({"error": "search_failed", "message": str(e)}), 500


# -----------------------------
# PIPELINE D'INDEXATION (PostgreSQL)
# -----------------------------
@app.route("/update_index", methods=["POST"])
def update_index():
    global ENGINE
    start_time = time.time()
    logger.info("🔄 DÉMARRAGE DE LA MISE À JOUR DE L'INDEX")

    try:
        # Release existing engine + GPU mem
        if ENGINE is not None:
            logger.info("🔓 Libération du moteur existant...")
            ENGINE = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)

        ppt_files = get_ppt_files_from_db()
        if not ppt_files:
            return jsonify({"error": "db_empty_or_unreachable"}), 400

        if OUTPUT_BASE_DIR.exists():
            logger.info(f"🧹 Nettoyage de {OUTPUT_BASE_DIR}...")
            try:
                shutil.rmtree(OUTPUT_BASE_DIR)
            except PermissionError:
                for item in OUTPUT_BASE_DIR.rglob("*"):
                    try:
                        if item.is_file():
                            item.unlink()
                    except Exception:
                        pass

        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        all_metadata = []
        for i, ppt_path in enumerate(ppt_files, 1):
            if ppt_path.exists():
                logger.info(f"➡️ [{i}/{len(ppt_files)}] Extraction : {ppt_path.name}")
                try:
                    meta = process_powerpoint(ppt_path, OUTPUT_BASE_DIR)
                    if meta:
                        all_metadata.extend(meta)
                except Exception as e:
                    logger.error(f"❌ Erreur extraction {ppt_path.name} : {e}")
            else:
                logger.error(f"❌ Fichier introuvable : {ppt_path}")

        if not all_metadata:
            return jsonify({"error": "no_metadata_extracted"}), 400

        meta_path = OUTPUT_BASE_DIR / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        logger.info("🧠 Calcul des embeddings (DINOv2)...")
        computer = EmbeddingComputer(model_name=MODEL_NAME)
        embeddings, valid_meta = computer.compute_batch(
            all_metadata,
            metadata_path=str(meta_path),
            images_root=str(IMAGES_DIR),
        )

        save_embeddings(embeddings, valid_meta, str(OUTPUT_BASE_DIR), MODEL_NAME)

        logger.info("🏗️ Construction de l'index FAISS...")
        build_faiss_index(
            embeddings_path=str(OUTPUT_BASE_DIR / f"embeddings_{MODEL_NAME}.npy"),
            output_dir=str(OUTPUT_BASE_DIR),
            model_name=MODEL_NAME,
        )

        config_path = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
        load_engine(config_path)

        total_duration = time.time() - start_time
        logger.info(f"✅ INDEXATION TERMINÉE EN {total_duration:.2f}s")

        return (
            jsonify(
                {
                    "status": "success",
                    "files_processed": len(ppt_files),
                    "images_indexed": len(valid_meta),
                    "duration": f"{total_duration:.2f}s",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"💥 CRASH DE L'INDEXATION : {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Micrograph API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port number")
    args = parser.parse_args()

    # Optional load on direct run as well
    conf = OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json"
    if conf.exists():
        load_engine(str(conf))

    app.run(host=args.host, port=args.port, debug=False)
