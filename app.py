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

from dotenv import load_dotenv

import numpy as np
import requests
import torch
from flask import Flask, jsonify, request, send_from_directory, send_file
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
from pgvector import Vector

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
# GROQ CLIENT
# -----------------------------------------------------------------------------
HARDCODED_GROQ_API_KEY = "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"
groq_api_key = HARDCODED_GROQ_API_KEY or os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None


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

        message = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=5000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
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
def add_formatted_markdown_to_docx(doc: Document, markdown_content: str):
    """
    Parse markdown content and add it to the DOCX with proper formatting.
    Handles: headers, subheaders, tables, bullet lists, and inline formatting.
    """
    lines = markdown_content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip empty lines
        if not line.strip():
            i += 1
            continue
        
        # Handle H2 headers (## title)
        if line.startswith('## '):
            title = line[3:].strip()
            heading = doc.add_heading(title, level=1)
            heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in heading.runs:
                run.font.size = Pt(13)
                run.font.bold = True
            i += 1
            continue
        
        # Handle H3 headers (### title)
        if line.startswith('### '):
            title = line[4:].strip()
            heading = doc.add_heading(title, level=2)
            heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in heading.runs:
                run.font.size = Pt(11)
                run.font.bold = True
            i += 1
            continue
        
        # Handle tables (markdown format: | header | header |)
        if line.startswith('|'):
            table_lines = [line]
            i += 1
            
            # Collect all table lines
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].rstrip())
                i += 1
            
            # Parse and create table
            try:
                # Extract header and separator
                if len(table_lines) >= 2:
                    header_line = table_lines[0]
                    headers = [h.strip() for h in header_line.split('|')[1:-1]]
                    
                    # Create table with num_cols x num_rows
                    rows_data = []
                    for row_line in table_lines[2:]:  # Skip header and separator
                        row_cells = [cell.strip() for cell in row_line.split('|')[1:-1]]
                        rows_data.append(row_cells)
                    
                    # Create table
                    if rows_data or headers:
                        table = doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
                        table.style = 'Light Grid Accent 1'
                        
                        # Add headers
                        header_cells = table.rows[0].cells
                        for col_idx, header in enumerate(headers):
                            cell = header_cells[col_idx]
                            paragraph = cell.paragraphs[0]
                            run = paragraph.add_run(header)
                            run.font.bold = True
                            run.font.size = Pt(10)
                        
                        # Add rows
                        for row_idx, row_data in enumerate(rows_data, 1):
                            row_cells = table.rows[row_idx].cells
                            for col_idx, cell_data in enumerate(row_data):
                                if col_idx < len(row_cells):
                                    cell = row_cells[col_idx]
                                    paragraph = cell.paragraphs[0]
                                    paragraph.text = cell_data
                                    paragraph.paragraph_format.space_before = Pt(6)
                                    paragraph.paragraph_format.space_after = Pt(6)
            except Exception as e:
                print(f"⚠️ Table parsing error: {e}")
                doc.add_paragraph(line)
            
            continue
        
        # Handle bullet/list items (*, +, -, followed by space)
        if line.startswith(('* ', '+ ', '- ')):
            # Collect consecutive list items
            list_items = []
            while i < len(lines):
                current = lines[i].rstrip()
                if current.startswith(('* ', '+ ', '- ')):
                    item_text = current[2:].strip()
                    # Check indentation level
                    indent_level = 0
                    if current.startswith(('  +', '  *', '  -')):
                        indent_level = 1
                    list_items.append((indent_level, item_text))
                    i += 1
                else:
                    break
            
            # Add list items to document
            for indent_level, item_text in list_items:
                p = doc.add_paragraph(item_text, style='List Bullet' if indent_level == 0 else 'List Bullet 2')
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
            
            continue
        
        # Default: add as regular paragraph with formatting
        paragraph = doc.add_paragraph()
        
        # Handle inline formatting: **bold**, *italic*, ***bold italic***
        text = line
        # Simple regex-like replacement for bold and italic
        parts = []
        last_end = 0
        
        # Process text for bold/italic
        j = 0
        while j < len(text):
            if text[j:j+3] == '***':
                # Add previous text
                if j > last_end:
                    run = paragraph.add_run(text[last_end:j])
                # Find closing ***
                close_idx = text.find('***', j + 3)
                if close_idx != -1:
                    bold_italic = text[j+3:close_idx]
                    run = paragraph.add_run(bold_italic)
                    run.bold = True
                    run.italic = True
                    j = close_idx + 3
                    last_end = j
                else:
                    j += 1
            elif text[j:j+2] == '**':
                # Add previous text
                if j > last_end:
                    run = paragraph.add_run(text[last_end:j])
                # Find closing **
                close_idx = text.find('**', j + 2)
                if close_idx != -1:
                    bold_text = text[j+2:close_idx]
                    run = paragraph.add_run(bold_text)
                    run.bold = True
                    j = close_idx + 2
                    last_end = j
                else:
                    j += 1
            elif text[j] == '*' or text[j] == '_':
                # Check if it's italic
                if (j > 0 and text[j-1] not in (' ', '\t')) or j == 0:
                    # Add previous text
                    if j > last_end:
                        run = paragraph.add_run(text[last_end:j])
                    # Find closing * or _
                    close_char = text[j]
                    close_idx = text.find(close_char, j + 1)
                    if close_idx != -1 and close_idx > j + 1:
                        italic_text = text[j+1:close_idx]
                        run = paragraph.add_run(italic_text)
                        run.italic = True
                        j = close_idx + 1
                        last_end = j
                    else:
                        j += 1
                else:
                    j += 1
            else:
                j += 1
        
        # Add remaining text
        if last_end < len(text):
            run = paragraph.add_run(text[last_end:])
        
        paragraph.paragraph_format.space_before = Pt(6)
        paragraph.paragraph_format.space_after = Pt(6)
        
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
                FROM public.fiches_ADN_matieres
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
                FROM public.fiches_ADN_matieres
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
                FROM public.fiches_ADN_matieres
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
        # Call Groq API
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
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


def generate_application_analysis_docx(fiche_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> str:
    """
    Generate a formatted DOCX document with application analysis.
    
    Returns the filepath to the generated DOCX file.
    """
    from datetime import datetime
    
    doc = Document()
    doc.default_paragraph_style.font.name = 'Arial'
    doc.default_paragraph_style.font.size = Pt(11)
    
    # Title
    title = doc.add_paragraph()
    title_run = title.add_run(f"Analyse d'usage du {fiche_data.get('nom_matiere', 'Matière')} {fiche_data.get('reference', 'N/A')}")
    title_run.font.size = Pt(14)
    title_run.font.bold = True
    
    # Section 1: Quick Reading
    doc.add_heading("1) Lecture rapide du matériau", level=2)
    doc.add_paragraph("Points clés issus de la fiche :")
    
    key_chars = analysis_data.get("material_summary", {}).get("key_characteristics", [])
    for char in key_chars:
        p = doc.add_paragraph(char, style='List Bullet')
    
    # Add interpretation
    doc.add_paragraph("➡️ Cela correspond à un matériau adapté aux applications suivantes :")
    domains = analysis_data.get("material_summary", {}).get("primary_domains", [])
    for domain in domains:
        p = doc.add_paragraph(f"aux {domain}", style='List Bullet')
    
    # Section 2: Main Application Domains
    doc.add_heading("2) Domaines d'application principaux", level=2)
    
    applications = analysis_data.get("applications", [])
    for idx, app in enumerate(applications, 1):
        letter = chr(64 + idx)  # A, B, C, etc.
        doc.add_heading(f"{letter}) {app.get('application_name', 'Application')}", level=3)
        
        doc.add_paragraph(f"Domain: {app.get('domain', 'N/A')} | Priority: {app.get('priority_level', 0)}")
        
        # Engagement process
        doc.add_heading("Engagement du matériau", level=4)
        process = app.get("engagement_process", {})
        process_desc = process.get("process_description", "")
        doc.add_paragraph(process_desc)
        
        steps = process.get("steps", [])
        if steps:
            for step in steps:
                step_text = f"{step.get('step_name', 'Step')}: {step.get('description', '')}"
                doc.add_paragraph(step_text, style='List Bullet')
        
        # Material role
        doc.add_heading("Rôle du matériau", level=4)
        material_role = process.get("material_role", "N/A")
        doc.add_paragraph(material_role)
        
        # Required properties
        doc.add_heading("Propriétés clés recherchées", level=4)
        props = app.get("required_properties", [])
        for prop in props:
            prop_text = f"{prop.get('property_name', 'Propriété')}: {prop.get('reason', '')}"
            doc.add_paragraph(prop_text, style='List Bullet')
        
        doc.add_paragraph()  # spacing
    
    # Section 3: Summary Table
    doc.add_heading("3) Tableau de synthèse", level=2)
    
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Light Grid Accent 1'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Application'
    hdr_cells[1].text = "Process d'engagement"
    hdr_cells[2].text = 'Rôle du matériau'
    hdr_cells[3].text = 'Propriétés clés'
    
    for app in applications:
        row_cells = table.add_row().cells
        row_cells[0].text = app.get('application_name', '')
        
        process = app.get("engagement_process", {})
        row_cells[1].text = process.get("process_description", "")[:100] + "..."
        row_cells[2].text = process.get("material_role", "")
        
        props = app.get("required_properties", [])
        prop_names = ", ".join([p.get('property_name', '') for p in props[:3]])
        row_cells[3].text = prop_names
    
    # Section 4: Strategic Opportunities
    doc.add_heading("4) Applications stratégiques hors cœur de métier", level=2)
    
    recommendations = analysis_data.get("strategic_recommendations", {})
    strategic = recommendations.get("strategic_expansion", [])
    
    doc.add_paragraph("Opportunités intéressantes :")
    for opp in strategic:
        text = f"{opp.get('opportunity', '')}: {opp.get('rationale', '')}"
        doc.add_paragraph(text, style='List Bullet')
    
    # Section 5: Strategic Reading
    doc.add_heading("5) Lecture stratégique pour votre groupe", level=2)
    
    doc.add_paragraph("Ce type de matériau :")
    doc.add_paragraph("est idéal pour :", style='List Bullet')
    for app in applications[:3]:
        doc.add_paragraph(app.get('application_name', ''), style='List Bullet 2')
    
    doc.add_paragraph("➡️ Il est parfaitement cohérent avec :")
    doc.add_paragraph("votre activité existante", style='List Bullet')
    doc.add_paragraph("vos projets de diversification", style='List Bullet')
    
    # Save to temp_docx folder
    from pathlib import Path
    temp_docx_dir = Path(__file__).resolve().parent / "temp_docx"
    temp_docx_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ref_safe = fiche_data.get("reference", "material").replace(" ", "_")
    filename = f"Analyse_{ref_safe}_{timestamp}.docx"
    filepath = temp_docx_dir / filename
    
    doc.save(str(filepath))
    
    return filename


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
                FROM public.fiches_ADN_matieres
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
                docx_filename = generate_application_analysis_docx(fiche_data, existing_analysis)
                download_url = f"/download_fiche_adn_docx/{docx_filename}"
                
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
                    SELECT fiche_adn_id FROM public.fiches_ADN_matieres
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
        docx_filename = generate_application_analysis_docx(fiche_data, analysis_result["analysis"])
        download_url = f"/download_fiche_adn_docx/{docx_filename}"
        
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
                
                docx_filename = generate_application_analysis_docx(fiche_data, analysis_json)
                docx_url = f"/download_fiche_adn_docx/{docx_filename}"
            
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
