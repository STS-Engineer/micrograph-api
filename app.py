from pathlib import Path
from threading import Thread

import requests
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from openai import OpenAI
from werkzeug.utils import secure_filename

# Imports from your local scripts
from extract_references_french_v3_3 import process_powerpoint
@@ -30,23 +32,56 @@
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
@@ -63,11 +98,9 @@ def cleanup_old_files(interval=1800): # Vérifie toutes les 30 minutes
                        print(f"⚠️ Erreur lors de la suppression de {f.name} : {e}")
        time.sleep(interval)


cleanup_thread = Thread(target=cleanup_old_files, daemon=True)
cleanup_thread.start()


# -----------------------------
# HELPERS
# -----------------------------
@@ -76,7 +109,6 @@ def load_engine(config_path: str):
    print(f"📄 Loading search engine: {config_path}")
    ENGINE = FrenchMicrographSearchEngine(config_path=config_path)


# -----------------------------
# AUTO-LOAD ENGINE (works under Gunicorn)
# -----------------------------
@@ -89,7 +121,6 @@ def load_engine(config_path: str):
        print(f"⚠️ Engine auto-load failed on import: {e}")
        print("   Use /update_index endpoint to build a new index")


# -----------------------------
# ROOT / HEALTH
# -----------------------------
@@ -105,10 +136,8 @@ def root():
        }
    ), 200


@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "ok",
@@ -117,7 +146,6 @@ def health():
        }
    ), 200


# -----------------------------
# IMAGE SERVING
# -----------------------------
@@ -126,78 +154,156 @@ def serve_image(filename):
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

@@ -222,39 +328,34 @@ def search():
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
@@ -296,7 +397,9 @@ def update_index():
        save_embeddings(embeddings, valid_metadata, str(OUTPUT_BASE_DIR), MODEL_NAME)

        embeddings_path = str(OUTPUT_BASE_DIR / f"embeddings_{MODEL_NAME}.npy")
        build_faiss_index(
            embeddings_path=embeddings_path, output_dir=str(OUTPUT_BASE_DIR), model_name=MODEL_NAME
        )

        config_path = str(OUTPUT_BASE_DIR / f"search_config_{MODEL_NAME}.json")
        load_engine(config_path)
@@ -308,7 +411,6 @@ def update_index():
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
