
from __future__ import annotations

import argparse
import io
import os
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Optional, List, Dict, Any
import re

from dotenv import load_dotenv

import numpy as np
import requests
import torch
from flask import Flask, jsonify, request, send_from_directory, send_file, url_for
from openai import OpenAI
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from werkzeug.utils import secure_filename
from groq import Groq
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pgvector.psycopg2 import register_vector
from flask import url_for

# Load environment variables from .env file
load_dotenv()

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

# Temporary DOCX files directory (cleaned every 1 hour)
DOCX_TEMP_DIR = BASE_DIR / "temp_docx"

OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOCX_TEMP_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# OPENAI CLIENT
# -----------------------------------------------------------------------------
HARDCODED_OPENAI_API_KEY = ""
openai_api_key = HARDCODED_OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key) if openai_api_key else None


# -----------------------------------------------------------------------------
# GROQ CLIENT WITH KEY ROTATION
# -----------------------------------------------------------------------------
GROQ_API_KEYS = [
    "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",  # Clé active
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
    "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",
]

# Index de la clé actuellement utilisée
current_groq_key_index = 0

# Initialiser le client avec la première clé
groq_api_key = GROQ_API_KEYS[current_groq_key_index] if GROQ_API_KEYS else os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


def rotate_groq_key():
    """Passe à la clé Groq suivante en cas d'échec d'authentification."""
    global current_groq_key_index, groq_client, groq_api_key
    
    current_groq_key_index = (current_groq_key_index + 1) % len(GROQ_API_KEYS)
    new_key = GROQ_API_KEYS[current_groq_key_index]
    
    print(f"🔄 Rotation vers la clé Groq #{current_groq_key_index + 1}")
    
    groq_api_key = new_key
    groq_client = Groq(api_key=new_key)
    
    return groq_client


def call_groq_with_retry(messages, model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=8000, response_format=None):
    """
    Appelle l'API Groq avec rotation automatique des clés en cas d'échec d'authentification.
    
    Args:
        messages: Liste des messages pour le chat
        model: Modèle à utiliser
        temperature: Température de génération
        max_tokens: Nombre maximum de tokens
        response_format: Format de réponse (ex: {"type": "json_object"})
    
    Returns:
        Réponse de l'API Groq
    
    Raises:
        Exception: Si toutes les clés ont échoué
    """
    if not groq_client:
        raise Exception("Groq client not initialized")
    
    attempts = 0
    max_attempts = len(GROQ_API_KEYS)
    
    while attempts < max_attempts:
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = groq_client.chat.completions.create(**kwargs)
            return response
            
        except Exception as e:
            error_message = str(e).lower()
            
            # Vérifier si c'est une erreur d'authentification ou de clé invalide
            if "authentication" in error_message or "invalid" in error_message or "unauthorized" in error_message or "401" in error_message:
                print(f"⚠️ Erreur d'authentification avec la clé #{current_groq_key_index + 1}: {e}")
                attempts += 1
                
                if attempts < max_attempts:
                    rotate_groq_key()
                    print(f"🔄 Tentative {attempts + 1}/{max_attempts} avec une nouvelle clé...")
                else:
                    raise Exception(f"❌ Toutes les clés Groq ({max_attempts}) ont échoué. Dernière erreur: {e}")
            else:
                # Autre type d'erreur, ne pas faire de rotation
                raise e
    
    raise Exception("Échec de l'appel à Groq après rotation de toutes les clés")


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
    """Deletes files in temp_uploads/ and temp_docx/ older than max_age_seconds."""
    while True:
        now = time.time()
        try:
            # Clean temp uploads
            for f in TEMP_UPLOAD_DIR.iterdir():
                if not f.is_file():
                    continue
                try:
                    age = now - f.stat().st_mtime
                    if age > max_age_seconds:
                        f.unlink(missing_ok=True)
                        print(f"🧹 Deleted old temp file: {f.name}")
                except Exception as e:
                    print(f"Error deleting {f.name}: {e}")
            
            # Clean temp DOCX files (remove files older than 1 hour)
            docx_max_age = 3600  # 1 hour
            for f in DOCX_TEMP_DIR.iterdir():
                if not f.is_file() or not f.suffix == ".docx":
                    continue
                try:
                    age = now - f.stat().st_mtime
                    if age > docx_max_age:
                        f.unlink(missing_ok=True)
                        print(f"🧹 Deleted old DOCX file: {f.name}")
                except Exception as e:
                    print(f"Error deleting {f.name}: {e}")
        
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        time.sleep(interval)


# Start cleanup thread
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
    query_vec = query_embedding.tolist()

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
# GROQ CONTENT GENERATION
# -----------------------------------------------------------------------------
def generate_fiche_adn_content_with_groq(
    fiche_data: Dict[str, Any],
    material_name: str,
    reference: str,
    type_matiere: str,
    specifications: List[Dict[str, Any]]
) -> str:
    """
    Generate detailed fiche ADN content using Groq API with structured format.
    Extracts data from the specifications dictionary structure.
    """
    if not groq_client:
        print("⚠️ Groq client not initialized. Using fallback content.")
        return generate_fallback_fiche_adn_content(material_name, reference, type_matiere, specifications)
    
    try:
        # Extract data from specifications structure
        datasheet_spec = None
        msds_spec = None
        lab_control_spec = None
        expert_notes_data = []
        fiches_data = []
        
        # If specifications is already a dict (from API), extract directly
        if isinstance(specifications, dict):
            specs_list = specifications.get("specifications", [])
            raw_expert_notes = specifications.get("expert_notes", [])
            fiches_data = specifications.get("fiches", [])
            
            # Extract note_json from expert_notes
            for note in raw_expert_notes:
                if isinstance(note, dict):
                    note_json = note.get("note_json", {})
                    if note_json:
                        expert_notes_data.append({
                            "expert_notes": note_json.get("expert_notes", ""),
                            "full_text": note_json.get("full_text", ""),
                            "magnification": note_json.get("magnification", ""),
                            "protocol": note_json.get("protocol", "")
                        })
        else:
            specs_list = specifications if isinstance(specifications, list) else []
        
        # Parse specification documents
        for spec in specs_list:
            if isinstance(spec, dict):
                source_type = spec.get("source_type", "").lower()
                donnees = spec.get("donnees", {})
                
                if "datasheet" in source_type:
                    datasheet_spec = spec
                elif "msds" in source_type:
                    msds_spec = spec
                elif "control" in source_type or "feuille" in source_type:
                    lab_control_spec = spec
        
        # Build comprehensive data structure for Groq
        prompt_data = {
            "material": {
                "nom_matiere": material_name,
                "reference": reference,
                "type_matiere": type_matiere
            },
            "datasheet": datasheet_spec.get("donnees", {}) if datasheet_spec else {},
            "msds": msds_spec.get("donnees", {}) if msds_spec else {},
            "lab_control": lab_control_spec.get("donnees", {}) if lab_control_spec else {},
            "expert_notes": expert_notes_data[:5] if expert_notes_data else [],  # Limit to 5
            "fiches": fiches_data
        }
        
        # Create comprehensive prompt (ENGLISH only)
        prompt = f"""Generate a COMPLETE and PROFESSIONAL MATERIAL DNA SHEET (FICHE ADN) in ENGLISH with the following strict structure.
Use ONLY the data provided in JSON. If a section lacks data, write "Not available".

⚠️ MANDATORY TRANSLATION RULE:
- ANY TEXT OR FIELD taken from the input data MUST be translated to ENGLISH
- This includes: material names, product designations, component names, descriptions, notes, observations
- Preserve technical identifiers (CAS numbers, references, chemical formulas)
- If field is in French, German, or any other language → TRANSLATE TO ENGLISH
- Examples:
  * "Graphite naturel" → "Natural graphite"
  * "Propriétés physicochimiques" → "Physicochemical properties"
  * "Matériel de sécurité" → "Safety equipment"

COMPLETE JSON DATA:
{json.dumps(prompt_data, ensure_ascii=False, default=str, indent=2)}

STRICT FORMATTING REQUIREMENTS:

I — IDENTITY & LOGISTICS
Extract from MSDS §1 and Datasheet:
- Material, Reference, Commercial designation
- Supplier: Name, Addresses (Head Office, Branches), Phone, Email, Website
- AVO Form, Supplier lot numbers
- Brand Lines
- Dates: Datasheet revision date, MSDS date
RULE: Only mention certificate checks (iO checks) if value ≠ "iO"

II — GENERAL PRODUCT CHARACTERISTICS
From MSDS §1 & §2:
- Product family
- Main chemical substance
- Official chemical name
- CAS # and EC #
- UN Transport Classification
- Synonyms
- COMPLETE list of components with CAS

III — CHEMICAL PROPERTIES - TRIPARTITE STRUCTURE
### III.1 QUANTIFIED PROPERTIES (Datasheet)
Table: Parameter | Min | Max | Unit

### III.2 DETAILED COMPOSITION (MSDS §2)
- Main components
- Heavy metals (Cd, Pb, Hg, Cr VI)
- Quartz ranges
- Important notes

### III.3 STABILITY & HAZARDS (MSDS §9, §10, §3)
- Thermal stability
- Decomposition products
- Incompatible materials
- Hazardous polymerization

## IV — PHYSICAL PROPERTIES
Consolidated table (MSDS §9 + Datasheet):
State | Appearance | Odor | Water solubility | Melting point | Specific weight | Electrical conductivity | Flash point | Bulk density

## V — LASER GRANULOMETRY
Particle size distribution (Datasheet):
Parameter | d10 (µm) | d50 (µm) | d90 (µm)

## VI — GRANULOMETRIC CONTROLS (LAB-CONTROL)
Table in ENGLISH:
Parameter | Test Method | Min | Max | Unit

## VII — EXPERT NOTES & OBSERVATIONS
From expert_notes:
- Examination protocol (magnifications, preparation)
- Particle morphology
- Process impact
- Critical attention points
- Safety recommendations

## VIII — STORAGE
Table from MSDS §7:
Adequate Conditions | Inadequate Conditions | Incompatibilities | Handling | Required Signaling | Temperature | Humidity

## IX — PACKAGING
From Datasheet:
- Packaging types (Bags, Big Bags, Pallets)
- Unit weights (standard bag, big bag)
- Transport recommendations
- Disposal instructions (MSDS §13)

## X — SAFETY — COMPLETE MSDS DATA
Extract from MSDS §1 to §17:

### X.1 CLASSIFICATION & IDENTIFICATION (§1, §2, §3)
- Hazard class
- NFPA Rating (Health, Flammability, Reactivity)
- Specific risks
- Exposure routes

### X.2 PROTECTION EQUIPMENT (§8)
- Exposure limits (ACGIH TLV, OSHA PEL)
- Respiratory equipment
- Protective gloves
- Eye protection
- Other PPE

### X.3 FIRST AID (§4)
Inhalation | Ingestion | Skin contact | Eye contact

### X.4 FIRE MEASURES (§5)
- Suitable extinguishing agents
- Hazardous combustion products
- Firefighter protection equipment

### X.5 SPILLS & CLEANUP (§6)
- Actions for small spills
- Actions for large spills
- Environmental precautions

### X.6 TOXICOLOGY (§11)
- Carcinogenicity
- Mutagenicity
- Reproductive toxicity
- LD50/LC50 data
- Specific effects (pneumoconiosis)

### X.7 ECOLOGY & DISPOSAL (§12, §13)
- Environmental effects
- Biodegradability
- Waste disposal method

OUTPUT FORMAT:
Use Markdown with:
- ## for main section headers
- ### for subsection headers
- Markdown tables for structured data
- Bullet lists for enumerations
- **bold** for keywords

QUALITY REQUIREMENTS:
✓ Accurate and complete data
✓ Professional format
✓ No hallucinations/invented data
✓ Rigid structure respected
✓ Ready for Word document integration
✓ ALL TEXT IN ENGLISH ONLY — Translate all input fields to English
✓ Preserve technical terms and identifiers (CAS numbers, reference codes)
✓ All descriptive content must be in English, never mix languages"""

        message = call_groq_with_retry(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=5000
        )
        
        content = message.choices[0].message.content if message.choices else ""
        
        if not content or len(content.strip()) < 100:
            print(f"⚠️ Groq returned empty or very short content: {len(content)} chars")
            print(f"Response: {content}")
            raise Exception("Groq returned insufficient content")
        
        print(f"✅ Groq generated {len(content)} chars of content")
        return content
    
    except Exception as e:
        print(f"❌ Groq generation failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't use fallback - raise the error so user knows Groq failed
        raise e


# -------------------------------------------------------
# MARKDOWN TO DOCX FORMATTING
# -------------------------------------------------------
def add_formatted_markdown_to_docx(doc: Document, markdown_text):
    """
    Parses markdown text and adds it to the DOCX document with basic formatting.
    Supports:
    - Headings (##, ###, ####)
    - Bold (**text**)
    - Unordered lists (* or -)
    - Simple tables (for the summary)
    """
    # Normalize line endings
    lines = markdown_text.strip().replace('\r\n', '\n').split('\n')
    
    in_table = False
    table = None
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue

        # Table handling
        if line.startswith('|') and '|' in line[1:]:
            if not in_table:
                # Start of a new table
                in_table = True
                header_line = line
                
                # Check for separator line
                if i + 1 < len(lines) and lines[i+1].strip().startswith('|--'):
                    separator_line = lines[i+1].strip()
                    
                    # Count columns from header
                    num_cols = len([h.strip() for h in header_line.split('|') if h.strip()])
                    
                    if num_cols > 0:
                        table = doc.add_table(rows=1, cols=num_cols)
                        table.style = 'Table Grid'
                        
                        # Populate header
                        hdr_cells = table.rows[0].cells
                        headers = [h.strip() for h in header_line.split('|') if h.strip()]
                        for j, header in enumerate(headers):
                            if j < num_cols:
                                hdr_cells[j].text = header
                        
                        i += 2 # Skip header and separator
                        continue
                else:
                    # Not a valid table, treat as plain text
                    in_table = False

            else: # Already in table
                if line.startswith('|'):
                    row_data = [cell.strip() for cell in line.split('|') if cell.strip()]
                    if table and len(row_data) == table.columns:
                        row_cells = table.add_row().cells
                        for j, cell_text in enumerate(row_data):
                            row_cells[j].text = cell_text
                        i += 1
                        continue
                    else:
                        # End of table or malformed row
                        in_table = False
                        table = None
                else:
                    # End of table
                    in_table = False
                    table = None

        # If not in a table or table processing is done for the line
        if not in_table:
            # Headings
            if line.startswith('### '):
                doc.add_heading(line[4:].strip(), level=3)
            elif line.startswith('## '):
                doc.add_heading(line[3:].strip(), level=2)
            elif line.startswith('# '):
                doc.add_heading(line[2:].strip(), level=1)
            
            # Unordered Lists
            elif line.startswith(('* ', '- ')):
                # Handle nested lists indicated by indentation
                indent_level = (len(line) - len(line.lstrip(' '))) // 2
                style = 'List Bullet'
                if indent_level > 0:
                    style = f'List Bullet {indent_level + 1}'
                
                p = doc.add_paragraph(line[2:].strip(), style=style)

            # Paragraphs with bold
            else:
                p = doc.add_paragraph()
                parts = re.split(r'(\*\*.*?\*\*)', line)
                for part in parts:
                    if part.startswith('**') and part.endswith('**'):
                        p.add_run(part[2:-2]).bold = True
                    elif part:
                        p.add_run(part)
            i += 1


def generate_fallback_fiche_adn_content(
    material_name: str,
    reference: str,
    type_matiere: str,
    specifications: List[Dict[str, Any]]
) -> str:
    """Generate fallback content if Groq fails."""
    content = f"""FICHE ADN - MATIÈRE

Nom: {material_name}
Référence: {reference}
Type: {type_matiere}

SPÉCIFICATIONS TECHNIQUES

"""
    if isinstance(specifications, list):
        for spec in specifications:
            if isinstance(spec, dict):
                content += f"• {spec.get('source_type', 'Donnée')}: {spec.get('donnees', 'N/A')}\n"
    
    content += """

RECOMMANDATIONS D'UTILISATION

Cette matière doit être manipulée selon les spécifications techniques ci-dessus.
Pour plus d'informations, veuillez consulter la documentation technique complète.
"""
    return content


def get_first_image_for_material(matiere_id: int) -> Optional[Image.Image]:
    """
    Retrieve the first image associated with a material from the database.
    Handles cross-platform paths (Windows backslashes vs Linux forward slashes).
    
    Args:
        matiere_id: Material ID
    
    Returns:
        PIL Image or None
    """
    conn = None
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT image_path
                FROM public.matiere_images
                WHERE matiere_id = %s
                LIMIT 1
                """,
                (matiere_id,)
            )
            result = cur.fetchone()
            
            if result and result.get("image_path"):
                image_path = result["image_path"]
                
                # Normalize path: convert Windows backslashes to forward slashes
                normalized_path = image_path.replace("\\", "/")
                filename = Path(normalized_path).name
                
                # Try to find the image file
                possible_paths = [
                    Path(normalized_path),  # Absolute or relative path from DB (normalized)
                    BASE_DIR / normalized_path,  # Relative to BASE_DIR
                    IMAGES_DIR / filename,  # embeddings_v7/images/filename
                    BASE_DIR / "output_v3" / "images" / filename,  # output_v3 location
                ]
                
                for file_path in possible_paths:
                    if file_path.exists():
                        try:
                            return Image.open(file_path).convert("RGB")
                        except Exception as e:
                            print(f"⚠️ Error loading image {file_path}: {e}")
        
        return None
    
    except Exception as e:
        print(f"⚠️ Error retrieving image: {e}")
        return None
    
    finally:
        if conn:
            conn.close()


def get_all_images_for_material(matiere_id: int, limit: int = 2) -> List[Dict[str, Any]]:
    """
    Retrieve images associated with a material from the database with magnification info.
    Handles cross-platform paths (Windows backslashes vs Linux forward slashes).
    
    Args:
        matiere_id: Material ID
        limit: Maximum number of images to retrieve (default: 2)
    
    Returns:
        List of dicts with keys: image_path, magnification, and image_obj (PIL Image)
    """
    conn = None
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    mi.id,
                    mi.image_path,
                    men.note_json
                FROM public.matiere_images mi
                LEFT JOIN public.matiere_expert_notes men ON men.matiere_image_id = mi.id
                WHERE mi.matiere_id = %s
                ORDER BY mi.id
                LIMIT %s
                """,
                (matiere_id, limit)
            )
            results = cur.fetchall()
            
            images_data = []
            for result in results:
                image_path = result.get("image_path")
                note_json = result.get("note_json") or {}
                magnification = note_json.get("magnification", "N/A") if isinstance(note_json, dict) else "N/A"
                
                # Try to find the image file
                image_obj = None
                if image_path:
                    # Normalize path: convert Windows backslashes to forward slashes
                    normalized_path = image_path.replace("\\", "/")
                    # Get just the filename from the path
                    filename = Path(normalized_path).name
                    
                    possible_paths = [
                        Path(normalized_path),  # Absolute or relative path from DB (normalized)
                        BASE_DIR / normalized_path,  # Relative to BASE_DIR (handles output_v3/images/...)
                        IMAGES_DIR / filename,  # embeddings_v7/images/filename
                        BASE_DIR / "output_v3" / "images" / filename,  # output_v3 location (cross-platform)
                    ]
                    
                    for file_path in possible_paths:
                        try:
                            if file_path.exists():
                                image_obj = Image.open(file_path).convert("RGB")
                                print(f"✅ Loaded image from: {file_path}")
                                break
                        except Exception as e:
                            print(f"⚠️ Error loading image {file_path}: {e}")
                    
                    if not image_obj:
                        print(f"⚠️ Image not found for: {image_path}")
                        print(f"   Normalized: {normalized_path}")
                        print(f"   Tried locations: {[str(p) for p in possible_paths]}")
                
                images_data.append({
                    "image_path": image_path,
                    "magnification": magnification,
                    "image_obj": image_obj
                })
            
            return images_data
        
    except Exception as e:
        print(f"⚠️ Error retrieving images: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        if conn:
            conn.close()


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
    check_db = request.args.get("check_db", "false").strip().lower() in {"1", "true", "yes"}
    db_ok = None
    db_error = None

    if check_db:
        try:
            conn = get_db_conn()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            conn.close()
            db_ok = True
        except Exception as e:
            db_ok = False
            db_error = str(e)

    return jsonify(
        {
            "status": "ok",
            "dino_loaded": DINO_MODEL is not None,
            "groq_configured": groq_client is not None,
            "openai_configured": client is not None,
            "db_ok": db_ok,
            "db_error": db_error,
        }
    ), 200


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
                if not client:
                    errors.append(f"{original_name}: OpenAI API key not configured. Use download_link instead.")
                    continue
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
# FICHE ADN - MATERIAL DNA SHEET
# -----------------------------------------------------------------------------
@app.route("/fiche_adn", methods=["GET"])
def get_fiche_adn():
    """
    Get the complete ADN (DNA) specifications sheet for a material.
    
    Query Parameters:
        - reference (str): Material reference (e.g., "6600135")
    
    Returns: Complete JSON specifications aggregating all fiches, specs, and expert notes
    """
    reference = request.args.get("reference", "").strip()
    
    if not reference:
        return jsonify({
            "success": False, 
            "error": "missing_parameters",
            "message": "The 'reference' query parameter is required"
        }), 400
    
    conn = None
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Query the fiches_ADN_matieres table
            cur.execute("""
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
                FROM public.fiches_adn_matieres
                WHERE UPPER(REPLACE(TRIM(reference), ' ', '')) = UPPER(REPLACE(%s, ' ', ''))
                LIMIT 1
            """, (reference,))
            
            result = cur.fetchone()
            
            if not result:
                return jsonify({
                    "success": False,
                    "error": "fiche_adn_not_found",
                    "message": f"No fiche ADN found for reference: {reference}"
                }), 404
            
            result_dict = dict(result)
            
            return jsonify({
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
                    "specifications": result_dict["specifications"]  # Complete JSON blob
                }
            }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "retrieval_failed",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


# Alternative endpoint: Get by matiere_id only (faster)
@app.route("/fiche_adn/<int:matiere_id>", methods=["GET"])
def get_fiche_adn_by_id(matiere_id):
    """
    Get the complete ADN specifications sheet for a material by matiere_id.
    
    Path Parameter:
        - matiere_id (int): The material ID
    
    Returns: Complete JSON specifications aggregating all fiches, specs, and expert notes
    """
    conn = None
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
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
                FROM public.fiches_adn_matieres
                WHERE matiere_id = %s
                LIMIT 1
            """, (matiere_id,))
            
            result = cur.fetchone()
            
            if not result:
                return jsonify({
                    "success": False,
                    "error": "fiche_adn_not_found",
                    "message": f"No fiche ADN found for matiere_id: {matiere_id}"
                }), 404
            
            result_dict = dict(result)
            
            return jsonify({
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
                    "specifications": result_dict["specifications"]  # Complete JSON blob
                }
            }), 200
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": "retrieval_failed",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


# GENERATE FICHE ADN DOCX
# -----------------------------------------------------------------------------
@app.route("/generate_fiche_adn_docx", methods=["GET"])
def generate_fiche_adn_docx():
    """
    Generate a DOCX file for the fiche ADN with material details and image.
    
    Query Parameters:
        - reference (str): Material reference (e.g., "6600135")
    
    Returns: DOCX file download or JSON error
    """
    reference = request.args.get("reference", "").strip()
    
    if not reference:
        return jsonify({
            "success": False,
            "error": "missing_parameters",
            "message": "The 'reference' query parameter is required"
        }), 400
    
    conn = None
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get fiche ADN data
            cur.execute("""
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
                FROM public.fiches_adn_matieres
                WHERE UPPER(REPLACE(TRIM(reference), ' ', '')) = UPPER(REPLACE(%s, ' ', ''))
                LIMIT 1
            """, (reference,))
            
            result = cur.fetchone()
            
            if not result:
                return jsonify({
                    "success": False,
                    "error": "fiche_adn_not_found",
                    "message": f"No fiche ADN found for reference: {reference}"
                }), 404
            
            result_dict = dict(result)
            matiere_id = result_dict["matiere_id"]
            material_name = result_dict["nom_matiere"]
            type_matiere = result_dict["type_matiere"]
            specifications = result_dict.get("specifications")
            
            # Parse specifications if it's a string JSON
            if isinstance(specifications, str):
                try:
                    specifications = json.loads(specifications)
                except:
                    specifications = {}
            
            # Ensure specifications is a dict (it should be from DB)
            if not isinstance(specifications, dict):
                specifications = {}
        
        # Generate content using Groq
        content = generate_fiche_adn_content_with_groq(
            result_dict,
            material_name,
            reference,
            type_matiere,
            specifications
        )
        
        # Create DOCX document
        doc = Document()
        
        # Add title
        title = doc.add_heading(f"MATERIAL DNA SHEET - {material_name}", level=1)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add reference info
        info_paragraph = doc.add_paragraph()
        info_run = info_paragraph.add_run(f"Reference: {reference}\n")
        info_run.bold = True
        info_paragraph.add_run(f"Type: {type_matiere}\n")
        info_paragraph.add_run(f"Generated on: {result_dict.get('date_creation', 'N/A')}")
        
        # Add a line break
        doc.add_paragraph()
        
        # Add generated content with proper formatting
        doc.add_heading("CONTENT", level=2)
        add_formatted_markdown_to_docx(doc, content)
        
        # Try to add images with magnification (limit to 2 different magnifications)
        images = get_all_images_for_material(matiere_id, limit=10)  # Get more to find 2 different magnifications
        if images:
            doc.add_page_break()
            
            # Add title for micrographies section
            title_paragraph = doc.add_heading("Examples of micrographie for a reference:", level=2)
            title_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            doc.add_paragraph()  # Add spacing
            
            # Find 2 images with different magnifications
            added_magnifications = set()
            images_to_add = []
            
            for img_data in images:
                if img_data["image_obj"] and len(images_to_add) < 2:
                    magnification = img_data.get("magnification", "N/A")
                    # Only add if we haven't added this magnification yet
                    if magnification not in added_magnifications:
                        images_to_add.append(img_data)
                        added_magnifications.add(magnification)
            
            # Add each image with its magnification (max 2)
            for idx, img_data in enumerate(images_to_add, 1):
                if img_data["image_obj"]:
                    # Add magnification heading
                    magnification = img_data.get("magnification", "N/A")
                    if magnification and magnification != "N/A":
                        mag_heading = doc.add_heading(f"Grossissement: {magnification}x", level=3)
                    else:
                        mag_heading = doc.add_heading(f"Grossissement: N/A", level=3)
                    mag_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    
                    # Save image temporarily and add to document
                    img_stream = io.BytesIO()
                    img_data["image_obj"].save(img_stream, format="PNG")
                    img_stream.seek(0)
                    
                    # Add image to document (max width: 5 inches)
                    try:
                        doc.add_picture(img_stream, width=Inches(5))
                        last_paragraph = doc.paragraphs[-1]
                        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception as e:
                        print(f"⚠️ Could not add image {idx} to document: {e}")
                    
                    doc.add_paragraph()  # Add spacing between images
        
        # Generate unique filename (timestamp + random to avoid collisions)
        timestamp = int(time.time())
        random_id = uuid.uuid4().hex[:8]
        filename = f"Fiche_ADN_{reference}_{timestamp}_{random_id}.docx"
        filepath = DOCX_TEMP_DIR / filename
        
        # Save document to temporary directory
        doc.save(str(filepath))
        
        print(f"✅ DOCX file saved to temp: {filepath}")
        print(f"📁 File will be automatically deleted after 1 hour")
        
        # Build absolute download URL
        # Get the host from request headers (for Azure deployment)
        host = request.host or os.getenv("API_HOST", "localhost:5000")
        # Check X-Forwarded-Proto header first (set by Azure load balancer)
        protocol = request.headers.get("X-Forwarded-Proto", request.scheme)
        # Force HTTPS for production domains
        if ".azurewebsites.net" in host or ".azure" in host:
            protocol = "https"
        absolute_download_url = f"{protocol}://{host}/download_fiche_adn_docx/{filename}"
        
        # Return JSON response with download URL
        return jsonify({
            "success": True,
            "file_name": filename,
            "download_url": f"/download_fiche_adn_docx/{filename}",
            "absolute_url": absolute_download_url,
            "expires_in": "1 hour",
            "message": f"DOCX file generated successfully for reference {reference}"
        }), 200
    
    except Exception as e:
        print(f"❌ Error generating fiche ADN: {e}")
        return jsonify({
            "success": False,
            "error": "generation_failed",
            "message": str(e)
        }), 500
    
    finally:
        if conn:
            conn.close()

# DOWNLOAD FICHE ADN DOCX
# -----------------------------------------------------------------------------
@app.route("/download_fiche_adn_docx/<filename>", methods=["GET"])
def download_fiche_adn_docx(filename):
    """
    Download a previously generated DOCX file.
    
    Path Parameters:
        - filename (str): Name of the DOCX file (e.g., "Fiche_ADN_6600323_1708367890.docx")
    
    Returns: DOCX file download or JSON error
    """
    try:
        # Security: Validate filename
        if not filename.endswith(".docx"):
            return jsonify({
                "success": False,
                "error": "invalid_file",
                "message": "Invalid file type"
            }), 400
        
        # Prevent path traversal attacks
        if ".." in filename or "/" in filename or "\\" in filename:
            return jsonify({
                "success": False,
                "error": "invalid_file",
                "message": "Invalid filename"
            }), 400
        
        # Build file path from temporary DOCX directory
        file_path = DOCX_TEMP_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            return jsonify({
                "success": False,
                "error": "file_not_found",
                "message": f"File not found: {filename}. File may have expired (1 hour retention)."
            }), 404
        
        # Return file
        return send_file(
            str(file_path),
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"❌ Error downloading DOCX: {e}")
        return jsonify({
            "success": False,
            "error": "download_failed",
            "message": str(e)
        }), 500

# POPULATE FICHES ADN TABLE
# -----------------------------------------------------------------------------
@app.route("/create_fiches_adn_table", methods=["POST"])
def create_fiches_adn_table():
    """
    Create the fiches_adn_matieres table if it doesn't exist.
    
    This should be called once before using populate_fiches_adn_table.
    
    Returns: Success message or error
    """
    try:
        conn = get_db_conn()
        
        with conn.cursor() as cur:
            # Create the table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.fiches_adn_matieres (
                    fiche_adn_id SERIAL PRIMARY KEY,
                    matiere_id INTEGER NOT NULL,
                    nom_matiere VARCHAR(255),
                    reference VARCHAR(100),
                    type_matiere VARCHAR(100),
                    specifications JSONB,
                    num_specifications INTEGER DEFAULT 0,
                    date_creation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    derniere_modification TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_matiere FOREIGN KEY (matiere_id) 
                        REFERENCES public.matieres(matiere_id) ON DELETE CASCADE,
                    CONSTRAINT unique_matiere_id UNIQUE (matiere_id)
                )
            """)
            
            # Create index on matiere_id for faster lookups
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_fiches_adn_matiere_id 
                ON public.fiches_adn_matieres(matiere_id)
            """)
            
            # Create index on reference for faster searches
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_fiches_adn_reference 
                ON public.fiches_adn_matieres(reference)
            """)
            
            conn.commit()
            
            # Check if table was created
            cur.execute("""
                SELECT COUNT(*) as count 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'fiches_adn_matieres'
            """)
            table_exists = cur.fetchone()[0] > 0
            
            return jsonify({
                "success": True,
                "message": "Table fiches_adn_matieres created successfully" if table_exists else "Table creation completed",
                "table_exists": table_exists
            }), 200
            
    except Exception as e:
        print(f"❌ Error creating fiches_adn_matieres table: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "table_creation_failed",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


@app.route("/populate_fiches_adn_table", methods=["POST"])
def populate_fiches_adn_table():
    """
    Populate/Update the fiches_adn_matieres table by aggregating data from:
    - matieres (main materials table)
    - fiches_matieres (technical sheets)
    - specifications (linked specifications)
    - matiere_expert_notes (expert notes with images)
    
    Process:
    1. Fetch all materials from database
    2. For each material, aggregate all related data into JSON
    3. Insert or update record in fiches_adn_matieres
    
    Returns: Summary of processed records
    """
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch all materials
            cur.execute("""
                SELECT matiere_id, nom_matiere, reference, type_matiere, 
                       date_creation, date_mise_a_jour
                FROM public.matieres
                ORDER BY matiere_id
            """)
            materials = cur.fetchall()
            
            if not materials:
                return jsonify({
                    "success": False,
                    "message": "No materials found in database"
                }), 404
            
            processed_count = 0
            updated_count = 0
            inserted_count = 0
            error_count = 0
            errors_details = []
            
            for material in materials:
                try:
                    matiere_id = material["matiere_id"]
                    nom_matiere = material["nom_matiere"]
                    reference = material["reference"]
                    type_matiere = material["type_matiere"]
                    
                    # Fetch fiches for this material
                    cur.execute("""
                        SELECT fiche_id, date_creation_fiche, derniere_modification
                        FROM public.fiches_matieres
                        WHERE matiere_id = %s
                        ORDER BY fiche_id DESC
                    """, (matiere_id,))
                    fiches = [dict(row) for row in cur.fetchall()]
                    
                    # Fetch all specifications for these fiches
                    specifications_list = []
                    for fiche in fiches:
                        cur.execute("""
                            SELECT spec_id, fiche_id, source_type, donnees, 
                                   date_creation, derniere_modification
                            FROM public.specifications
                            WHERE fiche_id = %s
                            ORDER BY spec_id
                        """, (fiche["fiche_id"],))
                        specifications_list.extend([dict(row) for row in cur.fetchall()])
                    
                    # Fetch expert notes with images
                    cur.execute("""
                        SELECT men.id, men.matiere_image_id, men.note_json, men.created_at,
                               mi.image_path
                        FROM public.matiere_expert_notes men
                        INNER JOIN public.matiere_images mi ON mi.id = men.matiere_image_id
                        WHERE mi.matiere_id = %s
                        ORDER BY men.created_at DESC
                    """, (matiere_id,))
                    expert_notes = [dict(row) for row in cur.fetchall()]
                    
                    # Aggregate everything into JSON structure
                    aggregated_data = {
                        "fiches": fiches,
                        "specifications": specifications_list,
                        "expert_notes": expert_notes,
                        "summary": {
                            "num_fiches": len(fiches),
                            "num_specifications": len(specifications_list),
                            "num_expert_notes": len(expert_notes)
                        }
                    }
                    
                    # Check if record already exists
                    cur.execute("""
                        SELECT fiche_adn_id FROM public.fiches_adn_matieres
                        WHERE matiere_id = %s
                    """, (matiere_id,))
                    existing = cur.fetchone()
                    
                    if existing:
                        # Update existing record
                        cur.execute("""
                            UPDATE public.fiches_adn_matieres
                            SET nom_matiere = %s,
                                reference = %s,
                                type_matiere = %s,
                                specifications = %s,
                                num_specifications = %s,
                                derniere_modification = CURRENT_TIMESTAMP
                            WHERE matiere_id = %s
                        """, (
                            nom_matiere,
                            reference,
                            type_matiere,
                            Json(aggregated_data),
                            len(specifications_list),
                            matiere_id
                        ))
                        updated_count += 1
                    else:
                        # Insert new record
                        cur.execute("""
                            INSERT INTO public.fiches_adn_matieres
                            (matiere_id, nom_matiere, reference, type_matiere, 
                             specifications, num_specifications, date_creation, derniere_modification)
                            VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (
                            matiere_id,
                            nom_matiere,
                            reference,
                            type_matiere,
                            Json(aggregated_data),
                            len(specifications_list)
                        ))
                        inserted_count += 1
                    
                    processed_count += 1
                    
                    # Commit after each material to ensure partial success
                    conn.commit()
                    
                except Exception as mat_error:
                    error_msg = str(mat_error)
                    print(f"❌ Error processing material {material.get('matiere_id')}: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    error_count += 1
                    errors_details.append({
                        "matiere_id": material.get('matiere_id'),
                        "reference": material.get('reference'),
                        "nom_matiere": material.get('nom_matiere'),
                        "error": error_msg
                    })
                    conn.rollback()
                    continue
            
            return jsonify({
                "success": True if error_count == 0 else False,
                "message": "Fiches ADN table populated successfully" if error_count == 0 else f"Population completed with {error_count} errors",
                "summary": {
                    "total_materials": len(materials),
                    "processed": processed_count,
                    "inserted": inserted_count,
                    "updated": updated_count,
                    "errors": error_count
                },
                "errors_details": errors_details[:10] if errors_details else []  # Return first 10 errors
            }), 200
            
    except Exception as e:
        print(f"❌ Error populating fiches_adn_matieres: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "population_failed",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


# VERIFY FICHES ADN TABLE
# -----------------------------------------------------------------------------
@app.route("/verify_fiches_adn_table", methods=["GET"])
def verify_fiches_adn_table():
    """
    Verify and analyze the fiches_adn_matieres table.
    
    Returns:
    - Total record count
    - Specifications statistics (total, average per material)
    - Sample records
    - Data quality metrics
    
    Returns: JSON with verification results
    """
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Count total records
            cur.execute("SELECT COUNT(*) as total FROM public.fiches_adn_matieres")
            total_count = cur.fetchone()["total"]
            
            if total_count == 0:
                return jsonify({
                    "success": True,
                    "message": "Table is empty",
                    "total_records": 0,
                    "statistics": {},
                    "samples": []
                }), 200
            
            # Get statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    SUM(num_specifications) as total_specifications,
                    AVG(num_specifications) as avg_specifications_per_material,
                    MIN(num_specifications) as min_specifications,
                    MAX(num_specifications) as max_specifications,
                    COUNT(CASE WHEN num_specifications = 0 THEN 1 END) as materials_without_specs,
                    COUNT(CASE WHEN specifications IS NOT NULL THEN 1 END) as materials_with_data
                FROM public.fiches_adn_matieres
            """)
            stats = dict(cur.fetchone())
            
            # Get sample records (first 5)
            cur.execute("""
                SELECT 
                    fiche_adn_id,
                    matiere_id,
                    nom_matiere,
                    reference,
                    type_matiere,
                    num_specifications,
                    date_creation,
                    derniere_modification,
                    specifications
                FROM public.fiches_adn_matieres
                ORDER BY derniere_modification DESC
                LIMIT 5
            """)
            samples = []
            for row in cur.fetchall():
                sample = dict(row)
                # Parse specifications JSON to show summary
                specs_data = sample.get("specifications", {})
                if isinstance(specs_data, str):
                    try:
                        specs_data = json.loads(specs_data)
                    except:
                        specs_data = {}
                
                sample["specifications_summary"] = {
                    "num_fiches": specs_data.get("summary", {}).get("num_fiches", 0) if isinstance(specs_data, dict) else 0,
                    "num_specifications": specs_data.get("summary", {}).get("num_specifications", 0) if isinstance(specs_data, dict) else 0,
                    "num_expert_notes": specs_data.get("summary", {}).get("num_expert_notes", 0) if isinstance(specs_data, dict) else 0
                }
                # Remove full specifications data from sample for brevity
                del sample["specifications"]
                samples.append(sample)
            
            # Get recent updates
            cur.execute("""
                SELECT 
                    fiche_adn_id,
                    nom_matiere,
                    reference,
                    derniere_modification
                FROM public.fiches_adn_matieres
                ORDER BY derniere_modification DESC
                LIMIT 10
            """)
            recent_updates = [dict(row) for row in cur.fetchall()]
            
            return jsonify({
                "success": True,
                "message": "Verification completed successfully",
                "total_records": total_count,
                "statistics": {
                    "total_records": int(stats["total_records"]),
                    "total_specifications": int(stats["total_specifications"] or 0),
                    "avg_specifications_per_material": float(stats["avg_specifications_per_material"] or 0),
                    "min_specifications": int(stats["min_specifications"] or 0),
                    "max_specifications": int(stats["max_specifications"] or 0),
                    "materials_without_specs": int(stats["materials_without_specs"] or 0),
                    "materials_with_data": int(stats["materials_with_data"] or 0)
                },
                "samples": samples,
                "recent_updates": recent_updates
            }), 200
            
    except Exception as e:
        print(f"❌ Error verifying fiches_adn_matieres: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "verification_failed",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


# GENERATE APPLICATIONS ANALYSIS
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
        if not client:
            return jsonify({"success": False, "error": "openai_not_configured", "message": "OpenAI API key not configured. Use download_link instead."}), 400
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
# APPLICATION ANALYSIS ENDPOINTS
# -----------------------------------------------------------------------------

def generate_application_analysis_with_llm(
    fiche_adn_data: Dict[str, Any],
    company_context: str = None
) -> Dict[str, Any]:
    """
    Generate comprehensive application analysis using LLM.
    
    Args:
        fiche_adn_data: Complete fiche ADN data (material specifications)
        company_context: AVOCarbon company context and product scope
        
    Returns:
        Structured analysis with applications, processes, properties, and opportunities
    """
    if not groq_client:
        return {
            "success": False,
            "error": "llm_not_configured",
            "message": "Groq client not initialized"
        }
    
    # Default company context if not provided
    if not company_context:
        company_context = """
AVOCarbon - Company Scope:
Core Business Areas:
1. Carbon brushes for electric motors (automotive, power tools, household appliances)
2. Brush-holder assemblies
3. Inductors and coils (chokes) for EMI filtering
4. Dynamic sealing joints (via Cyclam division)
5. Self-lubricating bearings and bushings
6. Friction rings, rotors, vanes for motors and pumps

Target Industries:
- Automotive (electric/hybrid vehicles, alternators, starters, auxiliary motors)
- Power tools
- Household appliances
- Industrial equipment
- Specific mobility (special vehicles, electric two-wheelers)

Technology Focus:
- Carbon and graphite composite materials
- Electrical conduction and EMI suppression
- Friction and wear management
- Self-lubrication systems
- Thermal management
"""
    
    # Extract key material properties
    material_name = fiche_adn_data.get("nom_matiere", "Unknown Material")
    reference = fiche_adn_data.get("reference", "N/A")
    type_matiere = fiche_adn_data.get("type_matiere", "N/A")
    specifications = fiche_adn_data.get("specifications", {})
    
    # Build comprehensive prompt
    prompt = f"""# Material Application Analysis Request

You are an industrial materials application engineer analyzing potential uses for a material.

## Material Information
- Name: {material_name}
- Reference: {reference}
- Type: {type_matiere}

## Material Specifications (Fiche ADN)
{json.dumps(specifications, indent=2, ensure_ascii=False)}

## Company Context
{company_context}

## Analysis Requirements

Please analyze this material and provide a comprehensive structured response with the following information:

1. **Main Application Domains**: List the primary application areas where this material is generally used

2. **Detailed Applications**: For each application, provide:
   a. Application name and category
   b. Industry/sector
   c. Is it within AVOCarbon's current scope? (core_business / strategic_opportunity / outside_scope)
   d. Priority level for AVOCarbon (1-5, where 5 is highest strategic importance)

3. **Manufacturing Engagement**: For each application, describe:
   a. Complete manufacturing/engagement process (step-by-step)
   b. Specific role of the material in this application
   c. Critical process parameters (temperature, pressure, time, etc.)

4. **Required Properties**: For each application, list:
   a. Critical material properties needed
   b. Performance specifications
   c. Why these properties matter for this application

5. **Strategic Opportunities**: Identify:
   a. Applications within AVOCarbon's current expertise that could be developed
   b. New market opportunities adjacent to current business
   c. Potential partnerships or technology transfers

## Response Format

Provide your response as a valid JSON object with this exact structure:

```json
{{
  "material_summary": {{
    "key_characteristics": ["list", "of", "key", "properties"],
    "primary_domains": ["domain1", "domain2"]
  }},
  "applications": [
    {{
      "application_name": "string",
      "application_category": "string (e.g., 'Carbon Brushes', 'Friction Materials')",
      "industry_sector": "string",
      "domain": "core_business | strategic_opportunity | outside_scope",
      "priority_level": 1-5,
      "engagement_process": {{
        "process_description": "detailed string",
        "steps": [
          {{
            "step_number": 1,
            "step_name": "string",
            "description": "string",
            "parameters": {{"temperature": "value", "pressure": "value"}}
          }}
        ],
        "material_role": "string describing the material's function"
      }},
      "required_properties": [
        {{
          "property_name": "string",
          "importance": "critical | important | beneficial",
          "target_value": "string or null",
          "reason": "why this property matters"
        }}
      ],
      "market_potential": {{
        "current_market_size": "string estimate or 'unknown'",
        "growth_trend": "growing | stable | declining | unknown",
        "competitive_advantage": "string or null"
      }}
    }}
  ],
  "strategic_recommendations": {{
    "within_scope": [
      {{
        "opportunity": "string",
        "rationale": "string",
        "development_effort": "low | medium | high"
      }}
    ],
    "strategic_expansion": [
      {{
        "opportunity": "string",
        "rationale": "string",
        "requirements": "string"
      }}
    ]
  }}
}}
```

Please provide only the JSON response, without any markdown formatting or code blocks.
"""
    
    try:
        # Call Groq API with retry
        response = call_groq_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert materials engineer specializing in industrial applications of carbon, graphite, and composite materials. You provide detailed, technically accurate analysis in structured JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        analysis_text = response.choices[0].message.content
        analysis_data = json.loads(analysis_text)
        
        return {
            "success": True,
            "analysis": analysis_data,
            "model_used": "llama-3.1-70b-versatile",
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing error: {e}")
        print(f"Raw response: {analysis_text[:500]}...")
        return {
            "success": False,
            "error": "json_parse_error",
            "message": str(e),
            "raw_response": analysis_text[:1000]
        }
    except Exception as e:
        print(f"⚠️ LLM generation error: {e}")
        return {
            "success": False,
            "error": "llm_generation_failed",
            "message": str(e)
        }


def generate_application_analysis_docx_with_llm(fiche_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> str:
    """
    Generate a formatted DOCX document with application analysis using LLM-generated content.
    
    Returns the filepath to the generated DOCX file.
    """
    if not groq_client:
        raise Exception("Groq client not initialized")

    # Build prompt for DOCX content generation
    prompt = f"""
You are a technical writer creating a professional DOCX report.
Based on the material data and analysis JSON provided, generate the full markdown content for the report.
Follow the requested structure exactly.

## Material Data
- Name: {fiche_data.get('nom_matiere', 'N/A')}
- Reference: {fiche_data.get('reference', 'N/A')}

## Analysis JSON
```json
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}
```

## Report Structure

### 1) Lecture rapide du matériau

**Points clés issus de la fiche :**
- [Liste des points clés, par exemple : Pureté carbone très élevée : 99,8 %, Faible teneur en cendres : 0,2 %, etc.]

**Interprétation :**
- [Interprétation détaillée, par exemple : Cela correspond à un graphite naturel de qualité industrielle fine, adapté à...]

### 2) Domaines d’application principaux

Pour chaque application (A, B, C, etc.) :

**A) [Nom de l'application]**

([Indication si l'application est cœur de métier, opportunité stratégique, etc. de votre groupe])

**Engagement du matériau**
- [Description détaillée du processus d'engagement du matériau, étape par étape]

**Rôle du graphite**
- [Description du rôle spécifique du matériau dans cette application]

**Propriétés clés recherchées**
- [Liste des propriétés clés, par exemple : Conductivité électrique, Faible friction, etc.]

[Répéter la structure ci-dessus pour chaque application (B, C, D, etc.)]

### 3) Tableau de synthèse

| Application | Process d’engagement | Rôle du graphite | Propriétés clés |
|-------------|----------------------|------------------|-----------------|
| [Application 1] | [Process 1] | [Rôle 1] | [Propriétés 1] |
| [Application 2] | [Process 2] | [Rôle 2] | [Propriétés 2] |
| ... | ... | ... | ... |

### 4) Applications stratégiques hors cœur de métier (potentiel de développement)

**Opportunités intéressantes**
- [Liste des opportunités, par exemple : Plastiques conducteurs pour électronique, Bagues autolubrifiantes pour pompes industrielles, etc.]

### 5) Lecture stratégique pour votre groupe

Ce type de graphite :
- [Résumé de la pertinence du matériau pour AVOCarbon, par exemple : est un graphite de formulation (pas un graphite structurel massif)]
- [Est idéal pour : balais carbone, composites autolubrifiants, plastiques techniques conducteurs]

➡️ Il est parfaitement cohérent avec :
- [Votre activité balais]
- [Vos projets bushings]
- [Vos pistes hors automobile]

Please provide only the complete markdown content for the document body.
"""

    try:
        response = call_groq_with_retry(
            messages=[
                {"role": "system", "content": "You are a technical writer that generates markdown content for reports."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=4000
        )
        
        markdown_content = response.choices[0].message.content
        
        # Create DOCX from markdown
        doc = Document()
        doc.add_heading(f"Analyse d'usage du {fiche_data.get('nom_matiere', 'Matière')} {fiche_data.get('reference', 'N/A')}", level=1)
        
        add_formatted_markdown_to_docx(doc, markdown_content)
        
        # Save to temp_docx folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ref_safe = fiche_data.get("reference", "material").replace(" ", "_")
        filename = f"Analyse_{ref_safe}_{timestamp}.docx"
        filepath = DOCX_TEMP_DIR / filename
        
        doc.save(str(filepath))
        
        return filename

    except Exception as e:
        print(f"⚠️ Error generating DOCX with LLM: {e}")
        raise e


@app.route("/generate_application_analysis", methods=["POST"])
def generate_application_analysis():
    """
    Generate application analysis for a material using LLM.
    
    Request body:
        {
            "reference": "material reference",
            "company_context": "optional custom company context",
            "save_to_db": true/false (default: true)
        }
    
    Returns:
        Structured application analysis with DOCX download link
    """
    data = request.get_json() or {}
    reference = data.get("reference", "").strip()
    company_context = data.get("company_context")
    save_to_db = data.get("save_to_db", True)
    
    if not reference:
        return jsonify({
            "success": False,
            "error": "missing_parameters",
            "message": "Parameter 'reference' is required"
        }), 400
    
    conn = None
    try:
        conn = get_db_conn()
        
        # Fetch fiche ADN
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    fiche_adn_id,
                    matiere_id,
                    nom_matiere,
                    reference,
                    type_matiere,
                    specifications
                FROM public.fiches_adn_matieres
                WHERE UPPER(REPLACE(TRIM(reference), ' ', '')) = UPPER(REPLACE(%s, ' ', ''))
                LIMIT 1
            """, (reference,))
            
            fiche_adn = cur.fetchone()
            
            if not fiche_adn:
                return jsonify({
                    "success": False,
                    "error": "fiche_adn_not_found",
                    "message": f"No fiche ADN found for reference: {reference}"
                }), 404
            
            fiche_data = dict(fiche_adn)
            
            # Check if analysis already exists for this reference
            cur.execute("""
                SELECT fiche_app_id, analysis_data FROM public.fiches_applications_matieres
                WHERE UPPER(REPLACE(TRIM(reference), ' ', '')) = UPPER(REPLACE(%s, ' ', ''))
                LIMIT 1
            """, (reference,))
            
            existing = cur.fetchone()
            if existing:
                # Return existing analysis instead of regenerating
                existing_dict = dict(existing)
                existing_analysis = existing_dict.get("analysis_data", {})
                
                # Generate DOCX if it doesn't exist
                docx_filename = generate_application_analysis_docx_with_llm(fiche_data, existing_analysis)
                # Force HTTPS for Azure deployment
                protocol = request.headers.get("X-Forwarded-Proto", request.scheme)
                if ".azurewebsites.net" in request.host or ".azure" in request.host:
                    protocol = "https"
                download_url = url_for('download_fiche_adn_docx', filename=docx_filename, _external=True, _scheme=protocol)
                
                return jsonify({
                    "success": True,
                    "message": "Analysis already exists for this reference",
                    "analysis": existing_analysis,
                    "fiche_app_id": existing_dict.get("fiche_app_id"),
                    "docx_filename": docx_filename,
                    "download_url": download_url,
                    "is_existing": True
                }), 200
        
        # Generate analysis using LLM
        analysis_result = generate_application_analysis_with_llm(fiche_data, company_context)
        
        if not analysis_result.get("success"):
            return jsonify(analysis_result), 500
        
        # Save to database (only if it was already unique)
        if save_to_db:
            analysis_data = analysis_result["analysis"]
            
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get fiche_adn_id from fiches_ADN_matieres
                cur.execute("""
                    SELECT fiche_adn_id FROM public.fiches_adn_matieres
                    WHERE matiere_id = %s
                    LIMIT 1
                """, (fiche_data["matiere_id"],))
                
                fiche_adn_row = cur.fetchone()
                fiche_adn_id = fiche_adn_row["fiche_adn_id"] if fiche_adn_row else None
                
                # Insert into fiches_applications_matieres (existing table)
                cur.execute("""
                    INSERT INTO public.fiches_applications_matieres
                    (matiere_id, fiche_adn_id, nom_matiere, reference, type_matiere, 
                     analysis_data, num_applications, date_creation, derniere_modification)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    RETURNING fiche_app_id
                """, (
                    fiche_data["matiere_id"],
                    fiche_adn_id,
                    fiche_data["nom_matiere"],
                    fiche_data["reference"],
                    fiche_data["type_matiere"],
                    Json(analysis_data),
                    len(analysis_data.get("applications", []))
                ))
                
                fiche_app_id = cur.fetchone()["fiche_app_id"]
                
                conn.commit()
                
                analysis_result["fiche_app_id"] = fiche_app_id
                analysis_result["saved_to_database"] = True
        
        # Generate DOCX
        docx_filename = generate_application_analysis_docx_with_llm(fiche_data, analysis_result["analysis"])
        # Force HTTPS for Azure deployment
        protocol = request.headers.get("X-Forwarded-Proto", request.scheme)
        if ".azurewebsites.net" in request.host or ".azure" in request.host:
            protocol = "https"
        download_url = url_for('download_fiche_adn_docx', filename=docx_filename, _external=True, _scheme=protocol)
        
        analysis_result["docx_filename"] = docx_filename
        analysis_result["download_url"] = download_url
        
        return jsonify(analysis_result), 200
        
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"⚠️ Error generating application analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "analysis_generation_failed",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


@app.route("/application_analysis", methods=["GET"])
def get_application_analysis():
    """
    Retrieve application analysis for a material by reference.
    
    Query parameters:
        - reference: Material reference code (required)
        - include_sessions: true/false (include all analysis sessions)
        - include_steps: true/false (include detailed process steps)
    
    Returns:
        Complete application analysis with DOCX download link
    """
    reference = request.args.get("reference", "").strip()
    include_sessions = request.args.get("include_sessions", "false").lower() == "true"
    include_steps = request.args.get("include_steps", "true").lower() == "true"
    
    if not reference:
        return jsonify({
            "success": False,
            "error": "missing_parameters",
            "message": "Query parameter 'reference' is required"
        }), 400
    
    conn = None
    try:
        conn = get_db_conn()
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get material info via reference
            cur.execute("""
                SELECT m.matiere_id, m.nom_matiere, m.reference, m.type_matiere
                FROM public.matieres m
                WHERE UPPER(REPLACE(TRIM(m.reference), ' ', '')) = UPPER(REPLACE(%s, ' ', ''))
                LIMIT 1
            """, (reference,))
            
            material = cur.fetchone()
            if not material:
                return jsonify({
                    "success": False,
                    "error": "material_not_found",
                    "message": f"Material with reference {reference} not found"
                }), 404
            
            material = dict(material)
            matiere_id = material["matiere_id"]
            
            # Get analyses from fiches_applications_matieres
            cur.execute("""
                SELECT 
                    fiche_app_id,
                    fiche_adn_id,
                    analysis_data,
                    num_applications,
                    date_creation,
                    derniere_modification
                FROM public.fiches_applications_matieres
                WHERE matiere_id = %s
                ORDER BY date_creation DESC
            """, (matiere_id,))
            
            fiches = [dict(row) for row in cur.fetchall()]
            
            # Get latest analysis
            latest_analysis = fiches[0] if fiches else None
            applications = []
            
            if latest_analysis:
                analysis_data = latest_analysis.get("analysis_data", {})
                applications = analysis_data.get("applications", [])
            
            # Process steps are already in the JSON structure
            if include_steps and applications:
                for app in applications:
                    process = app.get("engagement_process", {})
                    app["process_steps"] = process.get("steps", [])
            
            # Include all fiches if requested
            sessions = fiches if include_sessions else []
            
            # Build summary from JSON data
            summary = {
                "total_applications": len(applications),
                "by_domain": {},
                "by_priority": {"high": 0, "medium": 0, "low": 0},
                "total_analyses": len(fiches)
            }
            
            # Count by domain and priority from applications
            for app in applications:
                domain = app.get("domain", "unknown")
                summary["by_domain"][domain] = summary["by_domain"].get(domain, 0) + 1
                
                priority = app.get("priority_level", 0)
                if priority >= 4:
                    summary["by_priority"]["high"] += 1
                elif priority >= 2:
                    summary["by_priority"]["medium"] += 1
                else:
                    summary["by_priority"]["low"] += 1
            
            # Generate DOCX if analysis exists
            docx_url = None
            if latest_analysis:
                fiche_data = dict(material)
                fiche_data["nom_matiere"] = material.get("nom_matiere", "")
                analysis_json = latest_analysis.get("analysis_data", {})
                
                docx_filename = generate_application_analysis_docx_with_llm(fiche_data, analysis_json)
                # Force HTTPS for Azure deployment
                protocol = request.headers.get("X-Forwarded-Proto", request.scheme)
                if ".azurewebsites.net" in request.host or ".azure" in request.host:
                    protocol = "https"
                docx_url = url_for('download_fiche_adn_docx', filename=docx_filename, _external=True, _scheme=protocol)
            
            return jsonify({
                "success": True,
                "material": material,
                "applications": applications,
                "analysis_sessions": sessions if include_sessions else None,
                "summary": summary,
                "docx_download_url": docx_url
            }), 200
            
    except Exception as e:
        print(f"⚠️ Error retrieving application analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "retrieval_failed",
            "message": str(e)
        }), 500
    finally:
        if conn:
            conn.close()


# =============================================================================
# BLACK MIX CRUD ENDPOINTS
# =============================================================================

def build_black_mix_adn_snapshot(cur, black_mix_id, product_reference, mix_name):
    """
    Build complete JSON snapshot (ADN) of Black Mix for archiving/export/PDF generation.
    Uses existing database cursor to gather all related data.
    """
    
    # ── Black Mix header + revision history ──
    cur.execute(
        "SELECT document_revision_history FROM public.black_mixes WHERE id = %s",
        (black_mix_id,)
    )
    bm_row = cur.fetchone()
    revision_history = bm_row[0] if bm_row and bm_row[0] else None

    # ── Components ──
    cur.execute(
        """SELECT c.id, c.component_name, c.quantity_value, c.quantity_unit,
                  m.reference, m.nom_matiere, c.metadata
           FROM public.black_mix_components c
           JOIN public.matieres m ON c.matiere_id = m.matiere_id
           WHERE c.black_mix_id = %s
           ORDER BY c.id""",
        (black_mix_id,)
    )
    components = [
        {
            "id": r[0],
            "component_name": r[1],
            "quantity": float(r[2]) if r[2] is not None else None,
            "unit": r[3],
            "reference": r[4],
            "material_name": r[5],
            "metadata": r[6]
        }
        for r in cur.fetchall()
    ]

    # ── Process steps + step materials ──
    cur.execute(
        """SELECT s.id, s.step_order, s.step_name, s.machine_name, s.parameters,
                  ARRAY_AGG(m.reference ORDER BY m.reference) AS materials
           FROM public.black_mix_process_steps s
           LEFT JOIN public.black_mix_step_materials sm ON sm.process_step_id = s.id
           LEFT JOIN public.matieres m ON m.matiere_id = sm.matiere_id
           WHERE s.black_mix_id = %s
           GROUP BY s.id
           ORDER BY s.step_order""",
        (black_mix_id,)
    )
    process_steps = [
        {
            "step_order": r[1],
            "step_name": r[2],
            "machine": r[3],
            "parameters": r[4],
            "materials": list(r[5]) if r[5] and r[5][0] is not None else []
        }
        for r in cur.fetchall()
    ]

    step_materials = {}
    for step in process_steps:
        step_materials[str(step["step_order"])] = step["materials"]

    # ── Control plan ──
    cur.execute(
        """SELECT parameter_name, target_value, min_value, max_value, unit
           FROM public.black_mix_control_plan
           WHERE black_mix_id = %s
           ORDER BY parameter_name""",
        (black_mix_id,)
    )
    control_plan = [
        {
            "parameter_name": r[0],
            "target_value": float(r[1]) if r[1] is not None else None,
            "min_value": float(r[2]) if r[2] is not None else None,
            "max_value": float(r[3]) if r[3] is not None else None,
            "unit": r[4]
        }
        for r in cur.fetchall()
    ]

    return {
        "black_mix_id": black_mix_id,
        "product_reference": product_reference,
        "mix_name": mix_name,
        "status": "draft",
        "document_revision_history": revision_history,  # ← ajouté
        "created_at": datetime.now().isoformat(),
        "composition": components,
        "process_steps": process_steps,
        "step_materials": step_materials,
        "control_plan": control_plan,
        "snapshot_timestamp": datetime.now().isoformat()
    }


@app.route("/black-mix/validate-material/<reference>", methods=["GET"])
def validate_black_mix_material(reference):
    """Validate if a material reference exists in the database."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT matiere_id, nom_matiere, reference FROM public.matieres WHERE reference = %s",
                (reference,)
            )
            row = cur.fetchone()
            if row:
                return jsonify({
                    "reference": reference,
                    "exists": True,
                    "material_name": row[1],
                    "matiere_id": row[0]
                }), 200
            else:
                return jsonify({
                    "reference": reference,
                    "exists": False,
                    "material_name": None,
                    "matiere_id": None
                }), 200
    except Exception as e:
        logging.error(f"Validate material error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/submit", methods=["POST"])
def submit_black_mix():
    """Submit a complete Black Mix with components, process steps, and control plan."""

    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be JSON"
        }), 400

    data = request.get_json()

    product_reference = data.get("product_reference")
    mix_name = data.get("mix_name")
    components = data.get("components", [])
    process_steps = data.get("process_steps", [])
    control_plan = data.get("control_plan", [])
    document_revision_history = data.get("document_revision_history")

    # ------------------------------
    # Basic validation
    # ------------------------------
    if not product_reference or not mix_name:
        return jsonify({
            "success": False,
            "error": "product_reference and mix_name are required"
        }), 400

    if not process_steps:
        return jsonify({
            "success": False,
            "error": "At least one process_step is required"
        }), 400

    conn = psycopg2.connect(DB_DSN)

    try:
        with conn:
            with conn.cursor() as cur:

                # ------------------------------
                # Validate materials references
                # ------------------------------
                validation_errors = []

                for component in components:
                    ref = component.get("reference")
                    if not ref:
                        validation_errors.append("Component missing reference")
                        continue

                    cur.execute(
                        "SELECT matiere_id FROM public.matieres WHERE reference = %s",
                        (ref,)
                    )

                    if not cur.fetchone():
                        validation_errors.append(f"Invalid material reference: {ref}")

                if validation_errors:
                    return jsonify({
                        "success": False,
                        "validation_errors": validation_errors
                    }), 400

                # ------------------------------
                # Create Black Mix
                # ------------------------------
                cur.execute(
                    """
                    INSERT INTO public.black_mixes
                    (reference, name, status, created_at, document_revision_history)
                    VALUES (%s, %s, 'draft', NOW(), %s)
                    RETURNING id
                    """,
                    (
                        product_reference,
                        mix_name,
                        Json(document_revision_history) if document_revision_history else None
                    )
                )

                black_mix_id = cur.fetchone()[0]

                # ------------------------------
                # Insert components
                # ------------------------------
                for component in components:

                    cur.execute(
                        "SELECT matiere_id FROM public.matieres WHERE reference = %s",
                        (component["reference"],)
                    )

                    matiere_id = cur.fetchone()[0]

                    cur.execute(
                        """
                        INSERT INTO public.black_mix_components
                        (black_mix_id, matiere_id, component_name, quantity_value, quantity_unit, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            black_mix_id,
                            matiere_id,
                            component.get("component_name") or component["reference"],
                            component.get("quantity"),
                            component.get("unit", "phr"),
                            Json(component.get("metadata", {}))
                        )
                    )

                # ------------------------------
                # Insert process steps
                # ------------------------------
                for step in process_steps:

                    if not step.get("materials"):
                        raise ValueError(
                            f"Process step '{step.get('step_name')}' must contain at least one material"
                        )

                    cur.execute(
                        """
                        INSERT INTO public.black_mix_process_steps
                        (black_mix_id, step_order, step_name, machine_name, parameters)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            black_mix_id,
                            step.get("step_order"),
                            step.get("step_name"),
                            step.get("machine"),
                            Json(step.get("parameters", {}))
                        )
                    )

                    process_step_id = cur.fetchone()[0]

                    # ------------------------------
                    # Insert step-material relations
                    # ------------------------------
                    for ref in step.get("materials"):

                        cur.execute(
                            "SELECT matiere_id FROM public.matieres WHERE reference = %s",
                            (ref,)
                        )

                        mat_row = cur.fetchone()

                        if not mat_row:
                            raise ValueError(f"Invalid material reference in step: {ref}")

                        cur.execute(
                            """
                            INSERT INTO public.black_mix_step_materials
                            (process_step_id, matiere_id, created_at)
                            VALUES (%s, %s, NOW())
                            """,
                            (
                                process_step_id,
                                mat_row[0]
                            )
                        )

                # ------------------------------
                # Insert control plan
                # ------------------------------
                for param in control_plan:

                    cur.execute(
                        """
                        INSERT INTO public.black_mix_control_plan
                        (black_mix_id, parameter_name, target_value, min_value, max_value, unit)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            black_mix_id,
                            param.get("parameter_name"),
                            param.get("target_value"),
                            param.get("min_value"),
                            param.get("max_value"),
                            param.get("unit")
                        )
                    )

                # ------------------------------
                # Build ADN snapshot
                # ------------------------------
                adn_snapshot = build_black_mix_adn_snapshot(
                    cur,
                    black_mix_id,
                    product_reference,
                    mix_name
                )

                cur.execute(
                    """
                    INSERT INTO public.black_mix_adn
                    (black_mix_id, adn_text, version, created_at)
                    VALUES (%s, %s, 1, NOW())
                    RETURNING id
                    """,
                    (
                        black_mix_id,
                        Json(adn_snapshot)
                    )
                )

                adn_id = cur.fetchone()[0]

                return jsonify({
                    "success": True,
                    "message": f"Black Mix '{mix_name}' created successfully",
                    "black_mix_id": black_mix_id,
                    "product_reference": product_reference,
                    "adn": {
                        "id": adn_id,
                        "version": 1
                    }
                }), 200

    except Exception as e:
        conn.rollback()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        conn.close()

@app.route("/black-mix/list", methods=["GET"])
def list_black_mixes():
    """Get all Black Mixes."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, reference, name, status, created_at
                   FROM public.black_mixes
                   ORDER BY created_at DESC"""
            )
            rows = cur.fetchall()
            return jsonify({
                "success": True,
                "black_mixes": [
                    {
                        "id": r[0],
                        "product_reference": r[1],
                        "mix_name": r[2],
                        "status": r[3],
                        "created_at": r[4].isoformat() if r[4] else None
                    }
                    for r in rows
                ]
            }), 200
    except Exception as e:
        logging.error(f"List Black Mixes error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@app.route("/black-mix/<int:mix_id>", methods=["GET"])
def get_black_mix_details(mix_id):
    """Get complete details of a Black Mix."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, reference, name, status, created_at, document_revision_history
                   FROM public.black_mixes WHERE id = %s""",
                (mix_id,)
            )
            row = cur.fetchone()
            if not row:
                return jsonify({"success": False, "error": "Black Mix not found"}), 404

            result = {
                "id": row[0],
                "product_reference": row[1],
                "mix_name": row[2],
                "status": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "document_revision_history": row[5]
            }

            cur.execute(
                """SELECT c.id, c.component_name, c.quantity_value, c.quantity_unit,
                          m.reference, m.nom_matiere, c.metadata
                   FROM public.black_mix_components c
                   JOIN public.matieres m ON c.matiere_id = m.matiere_id
                   WHERE c.black_mix_id = %s""",
                (mix_id,)
            )
            result["components"] = [
                {
                    "id": r[0],
                    "component_name": r[1],
                    "quantity": float(r[2]) if r[2] is not None else None,
                    "unit": r[3],
                    "reference": r[4],
                    "material_name": r[5],
                    "metadata": r[6]
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                """SELECT s.id, s.step_order, s.step_name, s.machine_name, s.parameters,
                          ARRAY_AGG(m.reference ORDER BY m.reference) AS materials
                   FROM public.black_mix_process_steps s
                   LEFT JOIN public.black_mix_step_materials sm ON sm.process_step_id = s.id
                   LEFT JOIN public.matieres m ON m.matiere_id = sm.matiere_id
                   WHERE s.black_mix_id = %s
                   GROUP BY s.id
                   ORDER BY s.step_order""",
                (mix_id,)
            )
            result["process_steps"] = [
                {
                    "step_order": r[1],
                    "step_name": r[2],
                    "machine": r[3],
                    "parameters": r[4],
                    "materials": list(r[5]) if r[5] and r[5][0] is not None else []
                }
                for r in cur.fetchall()
            ]

            cur.execute(
                """SELECT parameter_name, target_value, min_value, max_value, unit
                   FROM public.black_mix_control_plan WHERE black_mix_id = %s""",
                (mix_id,)
            )
            result["control_plan"] = [
                {
                    "parameter_name": r[0],
                    "target_value": float(r[1]) if r[1] is not None else None,
                    "min_value": float(r[2]) if r[2] is not None else None,
                    "max_value": float(r[3]) if r[3] is not None else None,
                    "unit": r[4]
                }
                for r in cur.fetchall()
            ]

            return jsonify({"success": True, "black_mix": result}), 200

    except Exception as e:
        logging.error(f"Get Black Mix details error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# BLACK MIX ADN ENDPOINTS
# -----------------------------------------------------------------------------

@app.route("/black-mix/<int:mix_id>/adn", methods=["GET"])
def get_black_mix_adn(mix_id):
    """Retrieve the base ADN (DNA/snapshot) of a Black Mix for export/PDF/archiving."""
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            # Get ADN from database
            cur.execute(
                """SELECT id, black_mix_id, adn_text, version, created_at
                   FROM public.black_mix_adn
                   WHERE black_mix_id = %s
                   ORDER BY version DESC
                   LIMIT 1""",
                (mix_id,)
            )
            row = cur.fetchone()
            
            if not row:
                return jsonify({
                    "success": False,
                    "error": "ADN not found for this Black Mix"
                }), 404
            
            adn_id, black_mix_id, adn_text, version, created_at = row
            
            return jsonify({
                "success": True,
                "adn": {
                    "id": adn_id,
                    "black_mix_id": black_mix_id,
                    "version": version,
                    "created_at": created_at.isoformat() if created_at else None,
                    "snapshot": adn_text
                }
            }), 200

    except Exception as e:
        print(f"⚠️ Error getting ADN: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if conn:
            conn.close()


import json
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import jsonify

@app.route("/black-mix/<int:mix_id>/adn-enriched", methods=["GET"])
def get_black_mix_adn_enriched(mix_id):

    conn = psycopg2.connect(DB_DSN)

    try:
        with conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:

                # ---------------------------------------------------
                # 1. Black Mix identity
                # ---------------------------------------------------
                cur.execute("""
                    SELECT id, reference, name, status, document_revision_history
                    FROM black_mixes
                    WHERE id = %s
                """, (mix_id,))

                black_mix = cur.fetchone()

                if not black_mix:
                    return jsonify({"error": "Black mix not found"}), 404


                # ---------------------------------------------------
                # 2. Components + ADN specifications
                # ---------------------------------------------------
                cur.execute("""
                    SELECT
                        c.matiere_id,
                        c.component_name,
                        c.quantity_value,
                        c.quantity_unit,
                        m.reference,
                        m.nom_matiere,
                        m.type_matiere,
                        f.specifications
                    FROM black_mix_components c
                    JOIN matieres m
                        ON m.matiere_id = c.matiere_id
                    LEFT JOIN fiches_adn_matieres f
                        ON f.matiere_id = c.matiere_id
                    WHERE c.black_mix_id = %s
                """, (mix_id,))

                components = cur.fetchall()

                for c in components:
                    if not c["specifications"]:
                        c["specifications"] = "Information non disponible"


                # ---------------------------------------------------
                # 3. Process steps
                # ---------------------------------------------------
                cur.execute("""
                    SELECT
                        ps.id,
                        ps.step_order,
                        ps.step_name,
                        ps.machine_name,
                        ps.parameters
                    FROM black_mix_process_steps ps
                    WHERE ps.black_mix_id = %s
                    ORDER BY ps.step_order
                """, (mix_id,))

                process_steps = cur.fetchall()


                # ---------------------------------------------------
                # 4. Materials per step
                # ---------------------------------------------------
                cur.execute("""
                    SELECT
                        sm.process_step_id,
                        m.reference,
                        m.nom_matiere
                    FROM black_mix_step_materials sm
                    JOIN black_mix_process_steps ps
                        ON ps.id = sm.process_step_id
                    JOIN matieres m
                        ON m.matiere_id = sm.matiere_id
                    WHERE ps.black_mix_id = %s
                """, (mix_id,))

                step_materials = cur.fetchall()

                materials_by_step = {}

                for row in step_materials:
                    step_id = row["process_step_id"]

                    materials_by_step.setdefault(step_id, []).append({
                        "reference": row["reference"],
                        "nom_matiere": row["nom_matiere"]
                    })

                for step in process_steps:
                    step["materials"] = materials_by_step.get(step["id"], [])


                # ---------------------------------------------------
                # 5. Control plan
                # ---------------------------------------------------
                cur.execute("""
                    SELECT
                        parameter_name,
                        target_value,
                        min_value,
                        max_value,
                        unit
                    FROM black_mix_control_plan
                    WHERE black_mix_id = %s
                """, (mix_id,))

                control_plan = cur.fetchall()


                # ---------------------------------------------------
                # 6. Structured data for AI
                # ---------------------------------------------------
                data_for_ai = {
                    "black_mix_identity": {
                        "reference": black_mix["reference"],
                        "name": black_mix["name"],
                        "status": black_mix["status"],
                        "revision_history": black_mix["document_revision_history"]
                    },
                    "components": components,
                    "process_steps": process_steps,
                    "step_materials": step_materials,
                    "control_plan": control_plan
                }


                # ---------------------------------------------------
                # 7. YOUR PROMPT (unchanged)
                # ---------------------------------------------------
                prompt = f"""
Tu es un expert en formulation industrielle de matériaux carbone et graphite. 
Ton objectif est de générer un "Rapport Technique BLACK MIX ADN" en suivant STRICTEMENT la structure et le style du document de référence fourni, tout en intégrant une analyse approfondie de l'impact du processus de transformation.

### CONSIGNES DE RÉDACTION :
1. **AUCUNE HALLUCINATION** : N'invente aucune donnée, valeur numérique ou propriété. Si une information est manquante, indique "Information non disponible".
2. **STYLE PROFESSIONNEL** : Utilise un ton technique, précis et structuré (tableaux, listes à puces, sections numérotées).
3. **LANGUE** : Le rapport doit être rédigé en Français.

### STRUCTURE DU RAPPORT À RESPECTER :

#### 1. Introduction
Présente brièvement le rapport comme une référence de prompt structurée pour le mélange spécifique, garantissant la précision technique et la traçabilité.

#### 2. Identité et Vue d'ensemble du Black Mix
*   **2.1. Informations Générales** : Crée un tableau avec les paramètres : Référence Produit, Nom du Mix, Version ADN (si dispo), Statut, Système d'Origine (si dispo), Type de Document, Révision Actuelle.
*   **2.2. Historique des Révisions Clés** : Utilise les données de `revision_history` pour créer un tableau (Version, Date, Auteur, Description de la Modification). Analyse brièvement l'évolution si les données le permettent.

#### 3. Architecture et Processus du Black Mix
*   **3.1. Composition Structurelle et Phases** : Crée un tableau détaillé des composants (Code/Référence, Matériau, Quantité, Tolérance/Spécifications, Fonction, Phase). *Note : Déduis la phase (Sec, Humide, Final) selon la nature du composant et l'ordre d'introduction.*
*   **3.2. Gamme de Fabrication Détaillée** : Crée un tableau des étapes du processus (`process_steps`) incluant : Étape, Opération, Paramètres, Références Impliquées (via `step_materials`), Objectif.
*   **3.3. ANALYSE DE L'IMPACT DU PROCESSUS (NOUVEAU)** : 
    *   Analyse comment les paramètres de chaque étape influencent la qualité finale.
    *   Explique l'interaction entre machines et matériaux.
*   **3.4. Spécifications de Contrôle Final (Plan de Contrôle)** : Tableau basé sur `control_plan`.

#### 4. ADN Détaillé des Composants
Pour CHAQUE matière première listée dans les composants :
* Nom et Référence
* Type et Fonction
* Tableau des spécifications provenant de `specifications`.

#### 5. Synthèse de l'Identité Structurelle
Analyse finale + Note d'Expert.

### DONNÉES SOURCE (JSON) :
{json.dumps(data_for_ai, indent=2, ensure_ascii=False)}
"""

                # ---------------------------------------------------
                # 8. Call LLM
                # ---------------------------------------------------
                ai_response = call_groq_with_retry(prompt)


                # ---------------------------------------------------
                # 9. Return response
                # ---------------------------------------------------
                return jsonify({
                    "black_mix": black_mix,
                    "source_data": data_for_ai,
                    "ai_analysis": ai_response
                })


    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        conn.close()

# -----------------------------------------------------------------------------
# BLACK MIX COMBINED ADN (Black Mix ADN + Component fiches_adn_matieres)
# -----------------------------------------------------------------------------

@app.route("/black-mix/<int:mix_id>/adn-combined", methods=["GET"])
def get_black_mix_adn_combined(mix_id):
    """
    Return the full combined ADN: Black Mix snapshot + every component's
    fiches_adn_matieres (datasheet, MSDS, control-sheet specs) in one payload.
    This is the single source-of-truth endpoint for any downstream consumer
    (ChatGPT DOCX, BI dashboard, quality audit…).
    """
    conn = psycopg2.connect(DB_DSN)
    try:
        with conn.cursor() as cur:
            # ── 1. Black Mix header ──
            cur.execute(
                """SELECT id, reference, name, status, created_at, document_revision_history
                   FROM public.black_mixes WHERE id = %s""",
                (mix_id,)
            )
            bm = cur.fetchone()
            if not bm:
                return jsonify({"success": False, "error": "Black Mix not found"}), 404

            black_mix_info = {
                "id": bm[0], "product_reference": bm[1], "mix_name": bm[2],
                "status": bm[3],
                "created_at": bm[4].isoformat() if bm[4] else None,
                "document_revision_history": bm[5],
            }

            # ── 2. Base ADN snapshot ──
            cur.execute(
                """SELECT adn_text, version, created_at
                   FROM public.black_mix_adn
                   WHERE black_mix_id = %s ORDER BY version DESC LIMIT 1""",
                (mix_id,)
            )
            adn_row = cur.fetchone()
            if not adn_row:
                return jsonify({"success": False, "error": "ADN not found for this Black Mix"}), 404

            base_adn = adn_row[0]
            adn_version = adn_row[1]
            adn_created = adn_row[2].isoformat() if adn_row[2] else None

            # ── 3. Components with full ADN matières ──
            components_combined = []
            for comp in base_adn.get("composition", []):
                ref = comp.get("reference")
                entry = {
                    "reference": ref,
                    "material_name": comp.get("material_name", comp.get("component_name", "")),
                    "quantity": comp.get("quantity"),
                    "unit": comp.get("unit"),
                    "metadata": comp.get("metadata", {}),
                    "adn_matiere": None,
                }
                if ref:
                    cur.execute(
                        """SELECT fiche_adn_id, nom_matiere, material_name, reference,
                                  type_matiere, specifications, num_specifications
                           FROM public.fiches_adn_matieres
                           WHERE reference = %s
                           LIMIT 1""",
                        (ref,)
                    )
                    adn_row2 = cur.fetchone()
                    if adn_row2:
                        entry["adn_matiere"] = {
                            "fiche_adn_id": adn_row2[0],
                            "nom_matiere": adn_row2[1],
                            "material_name": adn_row2[2],
                            "type_matiere": adn_row2[4],
                            "specifications": adn_row2[5],
                            "num_specifications": adn_row2[6],
                        }
                components_combined.append(entry)

            # ── 4. Process steps with materials names ──
            process_steps_combined = []
            for step in base_adn.get("process_steps", []):
                mat_refs = step.get("materials", [])
                materials_detail = []
                for mref in mat_refs:
                    if not mref:
                        continue
                    # Find matching component info
                    mat_info = {"reference": mref}
                    for cc in components_combined:
                        if cc["reference"] == mref:
                            mat_info["material_name"] = cc["material_name"]
                            mat_info["quantity"] = cc["quantity"]
                            mat_info["unit"] = cc["unit"]
                            break
                    materials_detail.append(mat_info)
                process_steps_combined.append({
                    "step_order": step.get("step_order"),
                    "step_name": step.get("step_name"),
                    "machine": step.get("machine"),
                    "parameters": step.get("parameters"),
                    "material_references": mat_refs,
                    "materials_detail": materials_detail,
                })

            # ── 5. Control plan ──
            control_plan = base_adn.get("control_plan", [])

            return jsonify({
                "success": True,
                "black_mix": black_mix_info,
                "adn_version": adn_version,
                "adn_created_at": adn_created,
                "composition": components_combined,
                "process_steps": process_steps_combined,
                "control_plan": control_plan,
            }), 200

    except Exception as e:
        logging.error(f"Combined ADN error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # Start cleanup thread for temporary files (every 30 minutes)
    cleanup_thread = Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    print("🧹 Background cleanup task started (runs every 30 minutes)")

    app.run(host=args.host, port=args.port, debug=args.debug)
