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

import requests
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from openai import OpenAI
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Imports locaux
try:
    from extract_references_french_v3_3 import process_powerpoint
    from compute_embeddings import EmbeddingComputer, save_embeddings
    from build_faiss_index_proper import build_faiss_index
    from search_similar_french_v2 import FrenchMicrographSearchEngine
except ImportError as e:
    print(f"⚠️ Attention: Certains modules locaux sont manquants : {e}")

# -----------------------------
# CONFIGURATION DU LOGGING
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# -----------------------------
# CONFIGURATION GLOBALE
# -----------------------------
ENGINE = None
DB_URL = os.getenv("DATABASE_URL", "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA")

OUTPUT_BASE_DIR = Path("embeddings_v7")
IMAGES_DIR = OUTPUT_BASE_DIR / "images"
MODEL_NAME = "dinov2"

TEMP_UPLOAD_DIR = Path("temp_uploads")
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
client = OpenAI()

# -----------------------------
# UTILS & DATABASE
# -----------------------------
def get_ppt_files_from_db():
    paths = []
    conn = None
    try:
        logger.info("📡 Connexion à Azure PostgreSQL...")
        conn = psycopg2.connect(DB_URL, sslmode='require')
        cur = conn.cursor()
        cur.execute("SELECT file_path FROM powerpoint_files ")
        rows = cur.fetchall()
        paths = [Path(row[0].replace('"', '').strip()) for row in rows]
        cur.close()
        logger.info(f"✅ DB : {len(paths)} fichiers récupérés.")
    except Exception as e:
        logger.error(f"❌ Erreur DB : {e}")
    finally:
        if conn:
            conn.close()
    return paths

def load_engine(config_path: str):
    global ENGINE
    try:
        logger.info(f"⚙️ Chargement du moteur : {config_path}")
        if os.path.exists(config_path):
            ENGINE = FrenchMicrographSearchEngine(config_path=config_path)
            logger.info("✅ Moteur de recherche prêt !")
        else:
            logger.warning(f"⚠️ Fichier de configuration introuvable : {config_path}")
    except Exception as e:
        logger.error(f"⚠️ Échec du chargement du moteur : {e}")
        ENGINE = None

# -----------------------------
# NETTOYAGE AUTO (BACKGROUND)
# -----------------------------
def cleanup_old_files(interval=1800):
    while True:
        try:
            now = time.time()
            for f in TEMP_UPLOAD_DIR.glob("*"):
                if f.is_file() and now - f.stat().st_mtime > 7200:
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
@app.route("/health", methods=["GET"])
def health_check():
    """Route pour vérifier si l'API est en ligne et si le moteur est chargé."""
    return jsonify({
        "status": "ok", 
        "engine_ready": ENGINE is not None,
        "model": MODEL_NAME,
        "message": "Micrograph API is running"
    }), 200

@app.route("/temp_files/<path:filename>")
def serve_temp_image(filename):
    return send_from_directory(str(TEMP_UPLOAD_DIR), filename)

@app.route("/search", methods=["POST"])
def search():
    if ENGINE is None:
        return jsonify({"error": "engine_not_ready"}), 503
    
    data = request.get_json(silent=True) or {}
    top_k = data.get("top_k", 3)
    
    try:
        img = None
        if "temp_filename" in data:
            img_path = TEMP_UPLOAD_DIR / data["temp_filename"]
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
            else:
                return jsonify({"error": "file_not_found"}), 404
        elif "file_id" in data:
            file_response = client.files.content(data["file_id"])
            img = Image.open(io.BytesIO(file_response.read())).convert("RGB")
        else:
            return jsonify({"error": "no_input_provided"}), 400
            
        results = ENGINE.search_from_pil(img, top_k=top_k)
        return jsonify({"results": results}), 200
    except Exception as e:
        logger.error(f"Erreur lors de la recherche : {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# PIPELINE D'INDEXATION
# -----------------------------
@app.route("/update_index", methods=["POST"])
def update_index():
    global ENGINE
    start_time = time.time()
    logger.info("🔄 DÉMARRAGE DE LA MISE À JOUR DE L'INDEX")

    try:
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
                for item in OUTPUT_BASE_DIR.rglob('*'):
                    try: 
                        if item.is_file(): item.unlink()
                    except Exception: pass
        
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
            images_root=str(IMAGES_DIR)
        )

        save_embeddings(embeddings, valid_meta, str(OUTPUT_BASE_DIR), MODEL_NAME)
        
        logger.info("🏗️ Construction de l'index FAISS...")
        build_faiss_index(
            embeddings_path=str(OUTPUT_BASE_DIR / f"embeddings_{MODEL_NAME}.npy"),
            output_dir=str(OUTPUT_BASE_DIR),
            model_name=MODEL_NAME
        )

        config_path = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
        load_engine(config_path)

        total_duration = time.time() - start_time
        logger.info(f"✅ INDEXATION TERMINÉE EN {total_duration:.2f}s")
        
        return jsonify({
            "status": "success",
            "files_processed": len(ppt_files),
            "images_indexed": len(valid_meta),
            "duration": f"{total_duration:.2f}s"
        }), 200

    except Exception as e:
        logger.error(f"💥 CRASH DE L'INDEXATION : {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    # Parser pour permettre de passer l'hôte et le port en arguments comme dans vos logs
    parser = argparse.ArgumentParser(description="Run the Micrograph API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    conf = OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json"
    if conf.exists():
        load_engine(str(conf))

    app.run(host=args.host, port=args.port, debug=False)
