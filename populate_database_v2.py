#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction des micrographies depuis PowerPoints + insertion directe dans PostgreSQL.
Version V3 - Améliorée pour la gestion spatiale des grossissements et l'héritage des métadonnées.
CORRECTION V3.1:
  - Fix 1: extract_cokes_references_dict() — parsing ligne par ligne pour capturer les refs ( Vxxx )
  - Fix 2: extract_metadata_from_slide() — détection des refs de type "– RSxxx" (sans mot-clé "ref")
  - Fix 3: match_cokes_product_to_reference() — matching sans espaces pour CBH LPCS 60 / LPCS60
"""

import os
import re
import json
import io
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from transformers import AutoModel, AutoImageProcessor
from pptx import Presentation
from PIL import Image

import psycopg2
from psycopg2.extras import Json, RealDictCursor
from pgvector.psycopg2 import register_vector
from pgvector import Vector

DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"

# -------------------- SETUP DINOv2 --------------------
device = "cuda" if torch.torch.cuda.is_available() else "cpu"
print(f"🔧 Loading DINOv2-large on {device}...")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
print("✅ DINOv2 model loaded\n")

BASE_DIR = Path(__file__).resolve().parent

# -------------------- UTILITY FUNCTIONS --------------------
def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_reference(ref: str) -> str:
    """Clean and normalize a reference string for database matching."""
    if not ref:
        return ""
    ref = re.sub(r'^(ref|référence)\s*[:\-]?\s*', '', ref, flags=re.IGNORECASE)
    ref = re.sub(r'\s+', '', ref)
    return ref.strip().upper()

def compute_embedding_from_pil(image: Image.Image) -> np.ndarray:
    """Compute DINOv2 embedding (1024 dimensions)."""
    image = image.convert("RGB")
    inputs = dinov2_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding.astype("float32")

def _extract_nuance_from_text(text: str) -> Optional[str]:
    """Extract nuance like '477 00' or '52307'."""
    if not text:
        return None
    t = " ".join(str(text).split())
    m = re.search(r"\bNuance\s*[:\-]?\s*(\d{3})\s*(\d{2})\b", t, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = re.search(r"\b(\d{3})\s+(\d{2})\b", t)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = re.search(r"\b(\d{5})\b", t)
    if m:
        s = m.group(1)
        return f"{s[:3]} {s[3:]}"
    return None

def extract_detailed_comments(text: str) -> Optional[str]:
    """
    Extract detailed comments/description from a slide.
    """
    if not text:
        return None
    
    comments_match = re.search(
        r"Commentaires?\s*:\s*(.+)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    
    if comments_match:
        comments = comments_match.group(1).strip()
        
        for pattern in [r'\bTABLE\b', r'\bComposition\b', r'\bGrossissement\s+x\s*\d+']:
            match = re.search(pattern, comments, flags=re.IGNORECASE)
            if match:
                comments = comments[:match.start()].strip()
                break
        
        comments = re.sub(r'[ \t]+', ' ', comments)
        comments = re.sub(r'\n{3,}', '\n\n', comments)
        comments = comments.strip()
        
        if len(comments) > 20:
            return comments
    
    return None

def extract_magnifications_with_positions(slide) -> List[Dict]:
    """Extract all magnifications and their vertical positions."""
    mags = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text:
            match = re.search(r'[Gg]rossissement\s*[xX×]\s*(\d+)', shape.text)
            if match:
                mags.append({
                    "value": int(match.group(1)),
                    "text": shape.text.strip(),
                    "top": shape.top
                })
    return sorted(mags, key=lambda x: x["top"])

def extract_tables_from_slide(slide) -> List[List[List[str]]]:
    """Return all tables as: [table][row][cell_text]."""
    tables = []
    for shape in slide.shapes:
        if getattr(shape, "has_table", False):
            try:
                tbl = shape.table
                table_rows = []
                for r in tbl.rows:
                    row_cells = [(c.text or "").strip() for c in r.cells]
                    table_rows.append(row_cells)
                tables.append(table_rows)
            except Exception:
                continue
    return tables

def linearize_tables(tables: List[List[List[str]]]) -> str:
    """Convert extracted tables to a stable text form."""
    if not tables:
        return ""
    blocks = []
    for t_idx, t in enumerate(tables, start=1):
        lines = []
        for row in t:
            row2 = list(row)
            while row2 and row2[-1] == "":
                row2.pop()
            if not row2:
                continue
            lines.append(" | ".join(row2))
        if lines:
            blocks.append(f"TABLE_PPT_{t_idx}:\n" + "\n".join(lines))
    return "\n\n".join(blocks).strip()

# -------------------- COKES COMPARATIVE FILE HANDLING --------------------

def is_cokes_comparative_file(prs: Presentation) -> bool:
    """
    Détecte si le fichier PowerPoint est un fichier de comparaison de Cokes.
    """
    if len(prs.slides) < 2:
        return False
    
    slide1_text = []
    for shape in prs.slides[0].shapes:
        if hasattr(shape, "text") and shape.text.strip():
            slide1_text.append(shape.text.strip())
    slide1_full = " ".join(slide1_text)
    
    if "Coke" not in slide1_full:
        return False
    
    slide2_text = []
    for shape in prs.slides[1].shapes:
        if hasattr(shape, "text") and shape.text.strip():
            slide2_text.append(shape.text.strip())
    slide2_full = "\n".join(slide2_text)
    
    micrographie_count = slide2_full.count("Micrographie N°")
    
    return micrographie_count >= 3


def extract_cokes_comments_dict(prs: Presentation) -> Dict[str, str]:
    """
    Extrait tous les commentaires depuis le Slide 2 (commentaires centralisés).
    """
    if len(prs.slides) < 2:
        return {}
    
    slide = prs.slides[1]
    
    text_blocks = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            text_blocks.append(shape.text.strip())
    
    full_text = "\n".join(text_blocks)
    
    comments_dict = {}
    
    products = [
        "Coke MUCO Cyclam",
        "Coke FC 250",
        "Coke PDS 1183",
        "Coke CBH LPCS60",
        "Coke CBH LPCS 60",
        "Coke micronisé",
        "Coke MUCO 0-75µm",
        "Coke MUCO 0-75 µm",
        "Coke CARBOLEG FCB 97",
        "Coke CARBOLEG FCB97",
    ]
    
    for product in products:
        product_pattern = re.escape(product).replace(r"\ ", r"\s*")
        
        pattern = rf"{product_pattern}\s*(?:\([^)]+\))?\s*(?:–\s*Ref\s+\d+\s+\d+\s*)?\s*:\s*\tMicrographie N° \d+\s*\n(.+?)(?=\n\n[A-Z]|\Z)"
        
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        
        if match:
            comments = match.group(1).strip()
            comments = re.sub(r'[ \t]+', ' ', comments)
            comments = re.sub(r'\n{3,}', '\n\n', comments)
            comments = comments.strip()
            
            normalized_product = re.sub(r'\s+', ' ', product)
            comments_dict[normalized_product] = comments
    
    return comments_dict


# ==================== FIX 1 ====================
def extract_cokes_references_dict(prs: Presentation) -> Dict[str, str]:
    """
    Extrait les références depuis le Slide 1 (page de titre).

    FIX: Parsing ligne par ligne pour capturer à la fois:
      - "Product  \\tref XXXXXXX"  (référence numérique)
      - "Product  \\t( Vxxx )"     (code interne V679, V680, V681...)
      - "ProductSansRef\\nProductAvecRef  ref XXXXXXX"  (produit groupé sur la ligne précédente)
        → ex: "MUCO Cyclam" seul sur une ligne, "FC 250  ref 6600733" sur la suivante
        → les deux reçoivent 6600733 (look-ahead)
    
    Returns:
        Dict[product_name, reference]
    """
    if len(prs.slides) < 1:
        return {}
    
    slide = prs.slides[0]
    
    text_blocks = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            text_blocks.append(shape.text.strip())
    
    full_text = "\n".join(text_blocks)
    
    ref_dict = {}
    
    # First pass: parse each line into (product, ref_or_None)
    parsed = []
    for line in full_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Pattern 1: "Product  [tab] ref 6600xxx"
        m = re.match(r'^(.+?)\s+ref\s+(\d{5,7})\s*$', line, re.IGNORECASE)
        if m:
            product = m.group(1).strip()
            ref = m.group(2).strip()
            if not re.match(r'^coke\b', product, re.IGNORECASE):
                product = 'Coke ' + product
            parsed.append((product, ref))
            continue
        
        # Pattern 2: "Product  [tab] ( Vxxx )"  — internal vendor code
        m = re.match(r'^(.+?)\s+\(\s*(V\d+)\s*\)\s*$', line)
        if m:
            product = m.group(1).strip()
            ref = m.group(2).strip()
            if not re.match(r'^coke\b', product, re.IGNORECASE):
                product = 'Coke ' + product
            parsed.append((product, ref))
            continue
        
        # Pattern 3: Bare product name (no ref on this line) — skip headers/footers
        if re.search(r'(Micrographies|R&D|Date|\d{2}/\d{2}/\d{4})', line, re.IGNORECASE):
            continue
        parsed.append((line, None))
    
    # Second pass: fill ref_dict; if a line has no ref, inherit from the NEXT line's ref
    # (handles grouped products like "MUCO Cyclam" sharing "FC 250  ref 6600733")
    for i, (product, ref) in enumerate(parsed):
        if ref is None:
            # Look ahead for the next ref
            lookahead_ref = None
            for j in range(i + 1, len(parsed)):
                if parsed[j][1] is not None:
                    lookahead_ref = parsed[j][1]
                    break
            if lookahead_ref:
                if not re.match(r'^coke\b', product, re.IGNORECASE):
                    product = 'Coke ' + product
                ref_dict[product] = lookahead_ref
        else:
            ref_dict[product] = ref
    
    return ref_dict
# ==================== END FIX 1 ====================


# ==================== FIX 3 (part of Fix 1) ====================
def match_cokes_product_to_reference(product_name: str, ref_dict: Dict[str, str]) -> Optional[str]:
    """
    Match un nom de produit (depuis une slide d'images) aux références.

    FIX: Comparaison sans espaces pour gérer "LPCS 60" vs "LPCS60", et
         strip des suffixes ( Vxxx ) avant la comparaison.
    """
    product_name = re.sub(r'\s+', ' ', product_name).strip()
    
    # Exact match
    if product_name in ref_dict:
        return ref_dict[product_name]
    
    # Strip parenthetical suffixes (e.g. "( V679 )", "(0-75µm)")
    product_base = re.sub(r'\s*\([^)]+\)', '', product_name).strip()
    
    for ref_product, ref in ref_dict.items():
        ref_base = re.sub(r'\s*\([^)]+\)', '', ref_product).strip()
        
        # Compare without any whitespace, case-insensitive
        prod_norm = re.sub(r'\s+', '', product_base).lower()
        ref_norm  = re.sub(r'\s+', '', ref_base).lower()
        
        if prod_norm == ref_norm:
            return ref
        
        # Substring match (handles partial names)
        if ref_norm and (ref_norm in prod_norm or prod_norm in ref_norm):
            return ref
    
    return None
# ==================== END FIX 3 ====================


def match_cokes_product_to_comments(product_name: str, comments_dict: Dict[str, str]) -> Optional[str]:
    """
    Match un nom de produit (depuis une slide d'images) aux commentaires.
    """
    product_name = re.sub(r'\s+', ' ', product_name).strip()
    
    if product_name in comments_dict:
        return comments_dict[product_name]
    
    for comment_product, comments in comments_dict.items():
        product_base = re.sub(r'\s*\([^)]+\)', '', product_name).strip()
        comment_base = re.sub(r'\s*\([^)]+\)', '', comment_product).strip()
        
        product_normalized = re.sub(r'\s+', '', product_base).lower()
        comment_normalized = re.sub(r'\s+', '', comment_base).lower()
        
        if product_normalized == comment_normalized or \
           comment_base.lower() in product_base.lower() or \
           product_base.lower() in comment_base.lower():
            return comments
    
    return None


def extract_cokes_product_name_from_slide(slide) -> Optional[str]:
    """
    Extrait le nom du produit depuis une slide d'images.
    """
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            text = shape.text.strip()
            if "Coke" in text and "X " not in text and "x " not in text.lower():
                return text
    return None

def parse_avo_composition_from_tables(tables: List[List[List[str]]]) -> Optional[Dict[str, Any]]:
    """Extract composition table (AVO format) from real PPT tables."""
    for t in tables:
        rows = []
        for r in t:
            r2 = [(c or "").strip() for c in r]
            if any(c for c in r2):
                rows.append(r2)
        if len(rows) < 2:
            continue
        for idx in range(1, len(rows)):
            value_row = rows[idx]
            numeric_cells = [c for c in value_row if c and re.match(r"^[<>]?\s*\d+(?:[.,]\d+)?\s*$", c)]
            if len(numeric_cells) < 2:
                continue
            header_row = rows[0]
            pairs = []
            max_cols = min(len(header_row), len(value_row))
            for j in range(max_cols):
                h = (header_row[j] or "").strip()
                v = (value_row[j] or "").strip()
                if not h or not v or not re.match(r"^[<>]?\s*\d+(?:[.,]\d+)?\s*$", v):
                    continue
                pairs.append((h, v))
            if len(pairs) >= 2:
                elements = [h for h, _ in pairs]
                values = [v for _, v in pairs]
                return {"elements": elements, "values": values, "rows": [elements, values]}
    return None


# ==================== FIX 2 ====================
def extract_metadata_from_slide(slide, slide_number: int) -> Dict:
    """
    Extract metadata from a PowerPoint slide.

    FIX: Ajout d'un pattern de détection pour les références de type "– RSxxx"
         (sans mot-clé "ref"), utilisées dans graphite série 3.
    """
    metadata = {
        "slide_number": slide_number,
        "nuance": None,
        "grade": None,
        "product_name": None,
        "reference": None,
        "reference_raw": None,
        "magnifications": extract_magnifications_with_positions(slide),
        "comments": None,
        "composition": {},
        "full_text": "",
        "has_images": False,
        "is_title_page": False,
    }

    text_blocks = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text:
            text = shape.text.strip()
            if text:
                text_blocks.append(text)
                if not metadata["reference"]:
                    # Pattern 1: "ref XXXXX" or "référence XXXXX"
                    ref_match = re.search(
                        r'(?:ref|référence)\s*[:\-]?\s*([A-Z0-9\s]{4,15})',
                        text, re.IGNORECASE
                    )
                    if ref_match:
                        metadata["reference_raw"] = ref_match.group(1).strip()
                        metadata["reference"] = clean_reference(metadata["reference_raw"])
                    
                    # Pattern 2 (NEW): "– RSxxx" em-dash separated reference
                    # Handles "Graphite naturel Asbury #3478 – RS018" style
                    if not metadata["reference"]:
                        dash_ref = re.search(r'[–\-]\s*(RS\d+)\s*$', text)
                        if dash_ref:
                            metadata["reference_raw"] = dash_ref.group(1).strip()
                            metadata["reference"] = clean_reference(metadata["reference_raw"])
                
                nuance = _extract_nuance_from_text(text)
                if nuance and not metadata["nuance"]:
                    metadata["nuance"] = nuance

        if hasattr(shape, "image"):
            metadata["has_images"] = True

    metadata["full_text"] = "\n".join(text_blocks)
    
    if slide_number == 1 or (metadata["full_text"].count("ref") > 3 and not "Commentaires" in metadata["full_text"]):
        metadata["is_title_page"] = True
        metadata["reference"] = None
    
    if not metadata["is_title_page"]:
        comments = extract_detailed_comments(metadata["full_text"])
        if comments:
            metadata["comments"] = comments
    
    tables = extract_tables_from_slide(slide)
    if tables:
        composition = parse_avo_composition_from_tables(tables)
        if composition:
            metadata["composition"] = composition

    return metadata
# ==================== END FIX 2 ====================


def get_or_create_matiere_by_reference(cur, conn, entry: Dict) -> Optional[int]:
    """Get or create a matiere record by reference."""
    ref = entry.get("reference")
    if not ref:
        return None
    cur.execute("SELECT matiere_id FROM public.matieres WHERE reference = %s", (ref,))
    row = cur.fetchone()
    if row:
        return row[0]
    name = entry.get("product_name") or f"Matière {ref}"
    matiere_type = entry.get("type_matiere") or "Matière"
    cur.execute("""
        INSERT INTO public.matieres (nom_matiere, type_matiere, reference, date_creation, date_mise_a_jour)
        VALUES (%s, %s, %s, NOW(), NOW())
        RETURNING matiere_id
    """, (name, matiere_type, ref))
    new_id = cur.fetchone()[0]
    conn.commit()
    return new_id

def insert_image_and_notes(conn, cur, entry: Dict, source_file_id: int, matiere_id: int, embedding: np.ndarray):
    """Insert image and its associated notes into the database."""
    try:
        embedding_vector = Vector(embedding.tolist())
        cur.execute("""
            INSERT INTO public.matiere_images (matiere_id, image_path, embedding)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (matiere_id, entry["image_path"], embedding_vector))
        image_id = cur.fetchone()[0]

        note_json = {
            "expert_notes": entry.get("comments") or "",
            "magnification": entry.get("magnification"),
            "slide_number": entry.get("slide_number"),
            "source_file_id": source_file_id,
            "reference": entry.get("reference"),
            "composition": entry.get("composition"),
        }
        
        cur.execute("""
            INSERT INTO public.matiere_expert_notes (matiere_image_id, note_json, created_at)
            VALUES (%s, %s, NOW())
        """, (image_id, Json(note_json)))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"      ❌ Error inserting image: {e}")
        return False

def clear_old_data():
    """Clear old data from tables before reprocessing."""
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        
        print("🗑️  Clearing old data from database...")
        
        cur.execute("DELETE FROM public.matiere_expert_notes")
        print("   ✅ Cleared matiere_expert_notes")
        
        cur.execute("DELETE FROM public.matiere_images")
        print("   ✅ Cleared matiere_images")
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Old data cleared successfully")
        
        images_dir = BASE_DIR / "output_v3" / "images"
        if images_dir.exists():
            print("🗑️  Deleting old image files...")
            try:
                shutil.rmtree(images_dir)
                print("   ✅ Old image files deleted")
            except PermissionError:
                print("   ⚠️  Could not delete image directory (files may be locked)")
                print("   ℹ️  Continuing anyway - old files may be overwritten...")
        
        print()
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        import traceback
        traceback.print_exc()

def process_cokes_powerpoint(ppt_path: Path, output_dir: Path, file_id: int):
    """
    Traitement spécifique pour les fichiers de comparaison Cokes.
    """
    print(f"\n📊 Processing (Cokes format): {ppt_path.name}")
    
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    cur = conn.cursor()
    
    try:
        prs = Presentation(ppt_path)
        
        print("   Phase 1: Extraction des références...")
        ref_dict = extract_cokes_references_dict(prs)
        print(f"   → {len(ref_dict)} références trouvées: {list(ref_dict.items())}")
        
        print("   Phase 2: Extraction des commentaires...")
        comments_dict = extract_cokes_comments_dict(prs)
        print(f"   → {len(comments_dict)} produits avec commentaires")
        
        print("   Phase 3: Traitement des images...")
        
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(2, len(prs.slides)):
            slide = prs.slides[i]
            
            has_images = any(hasattr(shape, "image") for shape in slide.shapes)
            if not has_images:
                continue
            
            product_name = extract_cokes_product_name_from_slide(slide)
            if not product_name:
                print(f"   ⚠️  Slide {i+1}: Pas de nom de produit trouvé")
                continue
            
            comments = match_cokes_product_to_comments(product_name, comments_dict)
            reference = match_cokes_product_to_reference(product_name, ref_dict)
            
            matiere_id = get_or_create_matiere_by_reference(cur, conn, {
                "reference": reference,
                "product_name": product_name,
                "type_matiere": "Coke"
            })
            
            if not matiere_id:
                print(f"   ⚠️  Slide {i+1}: Could not create or find matiere record for '{product_name}' (ref={reference})")
                continue
            
            magnifications = extract_magnifications_with_positions(slide)
            
            img_count = 0
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    best_mag = None
                    if magnifications:
                        mags_above = [m for m in magnifications if m["top"] < shape.top]
                        if mags_above:
                            best_mag = mags_above[-1]["value"]
                        else:
                            best_mag = magnifications[0]["value"]
                    
                    image_bytes = shape.image.blob
                    filename = f"{ppt_path.stem}_s{i+1:03d}_i{img_count:02d}.png"
                    filepath = images_dir / filename
                    img = Image.open(io.BytesIO(image_bytes))
                    img.save(filepath, "PNG")
                    
                    embedding = compute_embedding_from_pil(img)
                    
                    entry = {
                        "image_path": str(filepath.relative_to(output_dir.parent)),
                        "magnification": best_mag,
                        "comments": comments,
                        "reference": reference,
                        "slide_number": i+1,
                        "composition": {}
                    }
                    
                    if insert_image_and_notes(conn, cur, entry, file_id, matiere_id, embedding):
                        com_status = f"✓ Com ({len(comments)} chars)" if comments else "✗ No com"
                        ref_status = f"Ref: {reference}" if reference else "No ref"
                        print(f"   ✅ Slide {i+1}: {product_name:35s} | {ref_status:15s} | {com_status}")
                    
                    img_count += 1
        
        print(f"✅ Finished processing {ppt_path.name}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cur.close()
        conn.close()

def process_powerpoint_wrapper(ppt_path: Path, output_dir: Path, file_id: int):
    """
    Smart router that detects file type and processes accordingly.
    """
    try:
        prs = Presentation(ppt_path)
        
        if is_cokes_comparative_file(prs):
            process_cokes_powerpoint(ppt_path, output_dir, file_id)
        else:
            process_standard_powerpoint(ppt_path, output_dir, file_id)
    except Exception as e:
        print(f"❌ Error detecting file type for {ppt_path.name}: {e}")
        import traceback
        traceback.print_exc()

def process_standard_powerpoint(ppt_path: Path, output_dir: Path, file_id: int):
    """
    Standard processing function for regular PowerPoint files.
    """
    print(f"\n📊 Processing (Standard format): {ppt_path.name}")
    
    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    cur = conn.cursor()
    
    try:

        prs = Presentation(ppt_path)
        all_slides_meta = []
        
        for i, slide in enumerate(prs.slides, 1):
            meta = extract_metadata_from_slide(slide, i)
            all_slides_meta.append(meta)
            comments_found = f"✓ Commentaires ({len(meta['comments'])} chars)" if meta["comments"] else "✗ Pas de commentaires"
            ref_found = f"Ref: {meta['reference']}" if meta["reference"] else "Aucune référence"
            print(f"   Slide {i}: {ref_found} [{comments_found}]")
            
        for i in range(len(all_slides_meta)):
            current = all_slides_meta[i]
            
            if current["has_images"]:
                if not current["reference"] and i > 0:
                    prev = all_slides_meta[i-1]
                    current["reference"] = prev["reference"]
                    current["reference_raw"] = prev["reference_raw"]
                    current["nuance"] = prev["nuance"]
                    if not current["comments"] and prev["comments"]:
                        current["comments"] = prev["comments"]
                        print(f"   🔄 Slide {i+1}: Métadonnées héritées de la slide {i}")
                    if not current["composition"] and prev["composition"]:
                        current["composition"] = prev["composition"]
                
                elif current["reference"] and not current["comments"]:
                    for j in range(i-1, -1, -1):
                        prev = all_slides_meta[j]
                        if prev["reference"] == current["reference"] and prev["comments"]:
                            current["comments"] = prev["comments"]
                            current["nuance"] = current["nuance"] or prev["nuance"]
                            current["composition"] = current["composition"] or prev["composition"]
                            print(f"   🔄 Slide {i+1}: Commentaires de {current['reference']} hérités de la slide {j+1}")
                            break

        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        for i, slide in enumerate(prs.slides):
            meta = all_slides_meta[i]
            if not meta["has_images"]:
                continue
                
            matiere_id = get_or_create_matiere_by_reference(cur, conn, meta)
            if not matiere_id:
                print(f"   ⚠️ Slide {i+1}: No reference found, skipping images.")
                continue

            img_count = 0
            for shape in slide.shapes:
                if hasattr(shape, "image"):
                    best_mag = None
                    if meta["magnifications"]:
                        mags_above = [m for m in meta["magnifications"] if m["top"] < shape.top]
                        if mags_above:
                            best_mag = mags_above[-1]["value"]
                        else:
                            best_mag = meta["magnifications"][0]["value"]

                    image_bytes = shape.image.blob
                    filename = f"{ppt_path.stem}_s{i+1:03d}_i{img_count:02d}.png"
                    filepath = images_dir / filename
                    img = Image.open(io.BytesIO(image_bytes))
                    img.save(filepath, "PNG")
                    
                    embedding = compute_embedding_from_pil(img)
                    
                    entry = {
                        "image_path": str(filepath.relative_to(output_dir.parent)),
                        "magnification": best_mag,
                        "comments": meta["comments"],
                        "reference": meta["reference"],
                        "slide_number": i+1,
                        "composition": meta["composition"]
                    }
                    
                    if meta["comments"]:
                        comment_preview = meta["comments"][:50] + "..." if len(meta["comments"]) > 50 else meta["comments"]
                        comments_status = f"✓ Commentaires: '{comment_preview}'"
                    else:
                        comments_status = "✗ Aucun commentaire"
                    
                    if insert_image_and_notes(conn, cur, entry, file_id, matiere_id, embedding):
                        print(f"   ✅ Slide {i+1}: Image {img_count} saved (Mag: x{best_mag}) [{comments_status}]")
                    
                    img_count += 1
                    
        print(f"✅ Finished processing {ppt_path.name}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    output = BASE_DIR / "output_v3"
    output.mkdir(exist_ok=True)
    
    clear_old_data()
    
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute("SELECT id, file_path FROM public.powerpoint_files")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if rows:
            print(f"Found {len(rows)} PowerPoint files to process from database\n")
            for row in rows:
                file_id = row[0]
                ppt_path = Path(row[1])
                if ppt_path.exists():
                    process_powerpoint_wrapper(ppt_path, output, file_id)
                else:
                    print(f"⚠️  File not found: {ppt_path}")
        else:
            print("⚠️  No PowerPoint files found in the database")
    except Exception as e:
        print(f"❌ Error retrieving files from database: {e}")
        import traceback
        traceback.print_exc()