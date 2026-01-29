import os
import argparse
import shutil
import json
import uuid
import time
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import torch

# Imports from your local scripts
from extract_references_french_v3_3 import process_powerpoint
from compute_embeddings import EmbeddingComputer, save_embeddings
from build_faiss_index_proper import build_faiss_index
from search_similar_french_v2 import FrenchMicrographSearchEngine

app = Flask(__name__)

# -----------------------------
# CONFIGURATION (Index/Search)  ✅ use Path (prevents Windows "\" path issues)
# -----------------------------
ENGINE = None
INPUT_PPT_DIR = Path("inputs")
OUTPUT_BASE_DIR = Path("embeddings_v7")
MODEL_NAME = "dinov2"

# -----------------------------
# CONFIGURATION (Temp Upload - FREE)
# -----------------------------
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

TMP_UPLOAD_DIR = Path(os.environ.get("TMP_UPLOAD_DIR", "/tmp/rfq_uploads"))
TMP_TTL_SECONDS = int(os.environ.get("TMP_TTL_SECONDS", "86400"))  # 24h by default

# Optional: upload size limit (16 MB)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def load_engine(config_path: str):
    """Loads or reloads the search engine into memory."""
    global ENGINE
    print(f"🔄 Loading search engine with config: {config_path}")
    ENGINE = FrenchMicrographSearchEngine(config_path=config_path)


def _ext(filename: str) -> str:
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def allowed_file(filename: str) -> bool:
    return _ext(filename) in ALLOWED_EXTENSIONS


# ✅ Add root endpoint: Render often pings "/" with HEAD
@app.route("/", methods=["GET", "HEAD"])
def root():
    return "OK", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "engine_loaded": ENGINE is not None,
            "model": MODEL_NAME,
            "tmp_upload_dir": str(TMP_UPLOAD_DIR),
            "tmp_ttl_seconds": TMP_TTL_SECONDS,
        }
    ), 200


# -----------------------------
# FREE TEMP FILE HOSTING
# -----------------------------
@app.route("/uploads/<path:filename>", methods=["GET"])
def serve_tmp_upload(filename):
    """Serve a temporary file stored on local disk."""
    return send_from_directory(str(TMP_UPLOAD_DIR), filename, as_attachment=False)


@app.route("/upload-image-temp", methods=["POST"])
def upload_image_temp():
    """
    JSON (single):
      { "download_link": "https://.../img.png" }

    JSON (multiple):
      { "images": [ {"download_link":"https://.../a.png"}, {"download_link":"https://.../b.jpg"} ] }

    Returns:
      { "status":"success", "paths":["/uploads/tmp_img_...png"], "errors":[] }
    """
    data = request.get_json(silent=True) or {}

    images = []
    if "download_link" in data:
        images = [data]
    else:
        images = data.get("images", [])

    if not images:
        return jsonify({"status": "error", "message": "Provide download_link or images[]"}), 400

    paths, errors = [], []

    for item in images:
        try:
            download_link = item.get("download_link")
            if not download_link:
                errors.append("Missing download_link")
                continue

            url_no_qs = download_link.split("?")[0]
            ext = _ext(url_no_qs) or "png"

            if ext not in ALLOWED_EXTENSIONS:
                errors.append(f"File type .{ext} not allowed (only jpg/jpeg/png)")
                continue

            r = requests.get(download_link, timeout=30)
            r.raise_for_status()

            ts = int(time.time())
            uid = uuid.uuid4().hex[:8]
            saved_name = f"tmp_img_{uid}_{ts}.{ext}"
            abs_path = TMP_UPLOAD_DIR / saved_name

            abs_path.write_bytes(r.content)
            paths.append(f"/uploads/{saved_name}")

        except Exception as e:
            errors.append(str(e))

    if not paths and errors:
        return jsonify(
            {"status": "error", "message": "All downloads failed", "paths": [], "errors": errors}
        ), 500

    return jsonify({"status": "success", "paths": paths, "errors": errors}), 200


@app.route("/cleanup-temp-local", methods=["POST"])
def cleanup_temp_local():
    """
    Deletes temp files older than TMP_TTL_SECONDS.
    (Call via external cron if needed.)
    """
    now = time.time()
    deleted = 0
    error_count = 0

    for p in TMP_UPLOAD_DIR.iterdir():
        try:
            if p.is_file():
                age = now - p.stat().st_mtime
                if age > TMP_TTL_SECONDS:
                    p.unlink()
                    deleted += 1
        except Exception:
            error_count += 1

    return jsonify({"status": "success", "deleted_count": deleted, "error_count": error_count}), 200


# -----------------------------
# INDEX PIPELINE
# -----------------------------
@app.route("/update_index", methods=["POST"])
def update_index():
    """
    Pipeline Route:
    1. Purge old output folder.
    2. Extract images/metadata from PPTs in /inputs.
    3. Compute DINOv2 embeddings.
    4. Build FAISS index.
    5. Refresh the Search Engine.
    """
    global ENGINE
    try:
        # 1. Reset Output Directory
        if OUTPUT_BASE_DIR.exists():
            print(f"🧹 Deleting old index at {OUTPUT_BASE_DIR}")
            shutil.rmtree(OUTPUT_BASE_DIR)
        OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

        # 2. Extract Data from PowerPoint
        input_path = INPUT_PPT_DIR
        ppt_files = list(input_path.glob("*.pptx")) + list(input_path.glob("*.ppt"))

        if not ppt_files:
            return jsonify({"error": "no_ppt_files_found", "path": str(input_path.resolve())}), 400

        print("📸 Step 1: Extracting images and metadata...")
        all_metadata = []
        for ppt_file in ppt_files:
            metadata = process_powerpoint(ppt_file, OUTPUT_BASE_DIR)
            all_metadata.extend(metadata)

        # Save interim metadata
        temp_meta_path = OUTPUT_BASE_DIR / "metadata.json"
        with open(temp_meta_path, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)

        # 3. Compute Embeddings
        print("🧬 Step 2: Computing embeddings...")
        computer = EmbeddingComputer(model_name=MODEL_NAME)
        embeddings, valid_metadata = computer.compute_batch(
            all_metadata,
            metadata_path=str(temp_meta_path),
            images_root=str(OUTPUT_BASE_DIR / "images"),
        )
        save_embeddings(embeddings, valid_metadata, str(OUTPUT_BASE_DIR), MODEL_NAME)

        # 4. Build FAISS Index
        print("🏗️ Step 3: Building FAISS index...")
        embeddings_path = str(OUTPUT_BASE_DIR / f"embeddings_{MODEL_NAME}.npy")
        build_faiss_index(
            embeddings_path=embeddings_path,
            output_dir=str(OUTPUT_BASE_DIR),
            model_name=MODEL_NAME,
        )

        # 5. Hot-Reload Engine
        new_config = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
        load_engine(new_config)

        # Optional: Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return jsonify(
            {
                "status": "success",
                "images_indexed": len(valid_metadata),
                "config": new_config,
            }
        ), 200

    except Exception as e:
        print(f"❌ Update failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------------
# SEARCH
# -----------------------------
@app.route("/search", methods=["POST"])
def search():
    """Similarity search using the current loaded engine."""
    if ENGINE is None:
        return jsonify({"error": "engine_not_loaded"}), 500

    if "file" not in request.files:
        return jsonify({"error": "missing_file"}), 400

    f = request.files["file"]
    try:
        img = Image.open(f.stream).convert("RGB")
        results = ENGINE.search_from_pil(img, top_k=1)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": "search_failed", "message": str(e)}), 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")

    # ✅ Bind to Render's assigned port if present
    default_port = int(os.environ.get("PORT", 8000))
    parser.add_argument("--port", type=int, default=default_port)

    args = parser.parse_args()

    # Ensure directories exist
    INPUT_PPT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-load existing index if it exists
    existing_config = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
    if os.path.exists(existing_config):
        try:
            load_engine(existing_config)
        except Exception as e:
            print(f"⚠️ Could not auto-load existing index: {e}")

    app.run(host=args.host, port=args.port)
