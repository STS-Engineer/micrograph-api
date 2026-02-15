#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction des micrographies depuis PowerPoints + insertion directe dans PostgreSQL.
Version CORRIGÉE - Liaison correcte images → matieres par référence.
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
device = "cuda" if torch.cuda.is_available() else "cpu"
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
    """
    Clean and normalize a reference string for database matching.
    Examples:
        "ref 6600 135" -> "6600135"
        "ref 6600135" -> "6600135"
        "6600 135" -> "6600135"
        "V494" -> "V494"
        "ref V 494" -> "V494"
    """
    if not ref:
        return ""
    
    # Remove common prefixes (ref, référence)
    ref = re.sub(r'^(ref|référence)\s*[:\-]?\s*', '', ref, flags=re.IGNORECASE)
    
    # Remove ALL whitespace
    ref = re.sub(r'\s+', '', ref)
    
    # Convert to uppercase for case-insensitive matching
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
    """Extract detailed comments/description from a slide."""
    if not text:
        return None
    comments_match = re.search(
        r"Commentaires\s*:(.*?)(?:\n\s*(?:Composition|TABLE|$))",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if comments_match:
        comments = comments_match.group(1).strip()
        comments = clean_text(comments)
        if len(comments) > 50:
            return comments
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        para_clean = clean_text(para)
        if len(para_clean) < 50:
            continue
        if re.match(r"^(Graphite|Coke|Nuance|Grossissement|Échelle)", para_clean, re.IGNORECASE):
            if len(para_clean) < 150:
                continue
        return para_clean
    return None

def extract_mag_scale_mapping(text: str) -> Dict[int, str]:
    """
    Extract magnification → scale mapping from metadata text.
    
    Examples:
        "x 200 ( échelle 100 µm ) – x 600 ( échelle 50 µm )"
        → {200: "100 µm", 600: "50 µm"}
    """
    if not text:
        return {}
    
    mapping = {}
    
    # Pattern: x 200 ( échelle 100 µm ) with flexible spacing
    pattern = r'[xX×]\s*(\d+)\s*\(\s*[ée]chelle\s*(\d+\s*[µμ]m)\s*\)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    for mag_str, scale in matches:
        mapping[int(mag_str)] = scale.strip()
    
    return mapping

def get_slide_magnification(slide) -> Optional[int]:
    """
    Extract magnification from slide header/text.
    
    Examples:
        "Grossissement x 200" → 200
        "Grossissement x600" → 600
    """
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text:
            match = re.search(r'[Gg]rossissement\s*[xX×]\s*(\d+)', shape.text)
            if match:
                return int(match.group(1))
    return None

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

_NUM_TOKEN = re.compile(r"^[<>]?\s*\d+(?:[.,]\d+)?\s*$")

def _is_number_like(s: str) -> bool:
    return bool(_NUM_TOKEN.match((s or "").strip()))

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
            numeric_cells = [c for c in value_row if c and _is_number_like(c)]
            if len(numeric_cells) < 2:
                continue
            header_row = rows[0]
            pairs = []
            max_cols = min(len(header_row), len(value_row))
            for j in range(max_cols):
                h = (header_row[j] or "").strip()
                v = (value_row[j] or "").strip()
                if not h:
                    continue
                if not v or not _is_number_like(v):
                    continue
                pairs.append((h, v))
            if len(pairs) >= 2:
                elements = [h for h, _ in pairs]
                values = [v for _, v in pairs]
                return {"elements": elements, "values": values, "rows": [elements, values]}
    return None

def extract_composition_table(text: str) -> Optional[Dict]:
    """Fallback: Extract composition table from plain text."""
    element_pattern = r"\b([A-Z][a-z]?)\b"
    lines = text.split("\n")
    for i, line in enumerate(lines):
        elements = re.findall(element_pattern, line)
        if len(elements) >= 3:
            if i + 1 < len(lines):
                values_line = lines[i + 1]
                values = re.findall(r"[<>]?\s*\d+(?:[.,]\d+)?", values_line)
                if len(values) >= len(elements):
                    return {
                        "elements": elements,
                        "values": [v.strip() for v in values[:len(elements)]],
                        "rows": [elements, [v.strip() for v in values[:len(elements)]]],
                    }
    return None

def composition_table_to_text(comp_table: Optional[Dict]) -> str:
    """Convert composition_table to readable text."""
    if not comp_table:
        return ""
    rows = comp_table.get("rows") or []
    if not rows:
        return ""
    lines = []
    for r in rows:
        lines.append(" | ".join([str(c).strip() for c in r]))
    return "\n".join(lines).strip()

def group_key(meta: Dict) -> Optional[str]:
    """Stable group key for entity aggregation."""
    if not meta:
        return None
    if meta.get("entity_type") == "Matière première":
        ref = (meta.get("reference") or "").strip()
        if ref:
            return f"MP|{ref}"
    return meta.get("entity_id")

def extract_metadata_from_slide(slide, slide_number: int) -> Dict:
    """Extract metadata from a PowerPoint slide."""
    metadata = {
        "slide_number": slide_number,
        "nuance": None,
        "grade": None,
        "product_name": None,
        "reference": None,
        "reference_raw": None,  # Original extracted text
        "magnification": None,
        "magnification_value": None,  # Numeric value for mapping
        "scale": None,
        "mag_scale_mapping": {},  # Magnification → scale mapping
        "comments": None,
        "description": None,
        "composition": {},
        "composition_table": None,
        "full_text": None,
        "has_images": False,
        "entity_type": None,
        "entity_id": None,
    }

    full_text_blocks = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text:
            full_text_blocks.append(shape.text)
        if hasattr(shape, "image"):
            metadata["has_images"] = True

    tables = extract_tables_from_slide(slide)
    tables_text = linearize_tables(tables)
    if tables_text:
        full_text_blocks.append(tables_text)

    raw_text = "\n".join(full_text_blocks)
    metadata["full_text"] = clean_text(raw_text)

    # Product name
    product_patterns = [
        r"Graphite\s+(Timrex|SFG|KS|BNL|Naturel|Artificiel)\s*[A-Z0-9]*",
        r"Graphite\s+(?:artificiel|naturel)\s+[A-Za-z0-9\s]+",
        r"Coke\s+[A-Z]{2,}",
        r"graphite\s+(?:artificiel|naturel)[\s\n]+[A-Za-z0-9\s]+",
    ]
    for pattern in product_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            product = re.sub(r"\s+", " ", match.group(0).strip())
            metadata["product_name"] = product
            break

    # Reference - IMPROVED EXTRACTION WITH MULTIPLE PATTERNS
    ref_patterns = [
        r"(?:ref|référence)\s*[:\-]?\s*(\d{4,}\s*\d{2,})",  # "ref 6600 135" or "ref 6600135"
        r"(?:ref|référence)\s*[:\-]?\s*([A-Z]\d+)",  # "ref V494"
        r"(?:ref|référence)\s*[:\-]?\s*([A-Z]\s*\d+)",  # "ref V 494"
        r"-\s*(\d{4,}\s*\d{2,})",  # "- 6600 135"
        r"–\s*(\d{4,}\s*\d{2,})",  # "– 6600 135"
        r"\b(\d{7})\b",  # Direct 7-digit like "6600135"
        r"\b([A-Z]\d{3,})\b",  # Alphanumeric like "V494"
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            ref_raw = match.group(1).strip()
            metadata["reference_raw"] = ref_raw
            # Clean the reference for database matching
            metadata["reference"] = clean_reference(ref_raw)
            break

    # Magnification
    mag_patterns = [
        r"Grossissement\s*[:\-]?\s*[xX×]?\s*(\d+)",
        r"[xX×]\s*(\d+)",
        r"(\d+)\s*[xX×]",
    ]
    for pattern in mag_patterns:
        match = re.search(pattern, raw_text)
        if match:
            mag_value = int(match.group(1))
            metadata["magnification"] = f"x{mag_value}"
            metadata["magnification_value"] = mag_value
            break

    # Extract magnification→scale mapping from comments
    metadata["mag_scale_mapping"] = extract_mag_scale_mapping(raw_text)

    # Scale - Try direct extraction first
    scale_patterns = [
        r"[ÉéEe]chelle\s*[:\-]?\s*(\d+\s*[µμ]?m)",
        r"(\d+\s*[µμ]m)",
    ]
    for pattern in scale_patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            metadata["scale"] = match.group(1).strip()
            break
    
    # If we have magnification and mapping, resolve scale from mapping
    if metadata["magnification_value"] and metadata["mag_scale_mapping"]:
        mapped_scale = metadata["mag_scale_mapping"].get(metadata["magnification_value"])
        if mapped_scale:
            metadata["scale"] = mapped_scale

    # Nuance
    nu = _extract_nuance_from_text(raw_text)
    if nu:
        metadata["nuance"] = nu

    # Comments
    detailed_comments = extract_detailed_comments(raw_text)
    if detailed_comments:
        metadata["comments"] = detailed_comments
        metadata["description"] = detailed_comments

    # Composition
    comp_table = parse_avo_composition_from_tables(tables)
    if not comp_table:
        comp_table = extract_composition_table(raw_text)
    if comp_table:
        metadata["composition_table"] = comp_table
        if "elements" in comp_table and "values" in comp_table:
            for elem, val in zip(comp_table["elements"], comp_table["values"]):
                metadata["composition"][elem] = val

    # Entity type / id
    if metadata.get("nuance"):
        metadata["entity_type"] = "Nuance"
        metadata["entity_id"] = metadata["nuance"]
    elif metadata.get("product_name"):
        metadata["entity_type"] = "Matière première"
        ref = (metadata.get("reference") or "").strip()
        metadata["entity_id"] = f"{metadata['product_name']}|{ref}".strip("|")
    elif metadata.get("grade"):
        metadata["entity_type"] = "Grade"
        metadata["entity_id"] = str(metadata["grade"]).strip()
    else:
        metadata["entity_type"] = "Inconnu"
        metadata["entity_id"] = None

    return metadata

# -------------------- DATABASE FUNCTIONS --------------------

def find_matiere_by_reference(cur, reference: str) -> Optional[int]:
    """
    Find a matiere_id by matching reference with flexible cleaning.
    Uses UPPER() and REPLACE() to handle case and whitespace variations.
    """
    if not reference:
        return None
    
    # Clean the reference using our standard function
    ref_clean = clean_reference(reference)
    
    # Query with flexible matching
    cur.execute("""
        SELECT matiere_id 
        FROM public.matieres 
        WHERE UPPER(REPLACE(REPLACE(reference, ' ', ''), '-', '')) = %s
        LIMIT 1
    """, (ref_clean,))
    row = cur.fetchone()
    
    if row:
        return row[0]
    
    return None

def create_new_matiere(cur, conn, entry: Dict) -> Optional[int]:
    """
    Create a new matiere entry when reference is not found.
    """
    reference = entry.get("reference")  # This is already cleaned
    reference_raw = entry.get("reference_raw") or reference
    
    if not reference:
        return None
    
    product_name = entry.get("product_name") or entry.get("nuance") or f"Material {reference}"
    material_name = product_name
    matiere_type = entry.get("entity_type") or "Matière première"
    
    try:
        cur.execute("""
            INSERT INTO public.matieres (
                nom_matiere, 
                material_name, 
                reference, 
                type_matiere, 
                date_creation
            )
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING matiere_id
        """, (product_name, material_name, reference, matiere_type))
        
        row = cur.fetchone()
        conn.commit()
        
        if row:
            print(f"      ✨ Created new matiere: {product_name} (ref: {reference}) -> matiere_id={row[0]}")
            return row[0]
        
    except psycopg2.IntegrityError as e:
        conn.rollback()
        print(f"      ⚠️  IntegrityError: {e}")
        # Try to find existing by reference
        existing_id = find_matiere_by_reference(cur, reference)
        if existing_id:
            print(f"      ℹ️  Found existing matiere_id={existing_id} for ref={reference}")
            return existing_id
    except Exception as e:
        conn.rollback()
        print(f"      ❌ Error creating matiere: {e}")
    
    return None

def get_or_create_matiere_by_reference(cur, conn, entry: Dict) -> Optional[int]:
    """
    Main function: Get matiere_id by reference, or create new if not found.
    Returns None if no reference available.
    """
    reference = entry.get("reference")
    
    if not reference:
        return None
    
    # Try to find existing matiere
    matiere_id = find_matiere_by_reference(cur, reference)
    
    if matiere_id:
        return matiere_id
    
    # Create new matiere
    return create_new_matiere(cur, conn, entry)

def get_or_create_powerpoint_file(cur, conn, ppt_path: Path) -> int:
    """Get or create PowerPoint file entry."""
    file_path_str = str(ppt_path.resolve())
    cur.execute("SELECT id FROM public.powerpoint_files WHERE file_path = %s", (file_path_str,))
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute("""
        INSERT INTO public.powerpoint_files (file_path, created_at)
        VALUES (%s, NOW())
        RETURNING id
    """, (file_path_str,))
    row = cur.fetchone()
    conn.commit()
    return row[0]

def insert_image_and_notes(conn, cur, entry: Dict, source_file_id: int, matiere_id: Optional[int], embedding: np.ndarray):
    """Insert image with embedding and metadata."""
    if not matiere_id:
        print(f"      ❌ Cannot insert image: no matiere_id")
        return False
    
    try:
        # Convert numpy array to pgvector Vector
        embedding_vector = Vector(embedding.tolist())
        
        # Insert image - ONLY foreign key, path, and embedding
        # material_name and reference come from matieres table via JOIN
        cur.execute("""
            INSERT INTO public.matiere_images (matiere_id, image_path, embedding)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (matiere_id, entry["image_path"], embedding_vector))
        image_id = cur.fetchone()[0]

        # Build metadata JSON
        note_json = {
            "expert_notes": entry.get("comments") or entry.get("description") or "",
            "magnification": entry.get("magnification"),
            "scale": entry.get("scale"),
            "composition": entry.get("composition"),
            "composition_table": entry.get("composition_table"),
            "full_text": entry.get("entity_full_text") or entry.get("full_text"),
            "slide_number": entry.get("slide_number"),
            "source_file_id": source_file_id,
            "product_name": entry.get("product_name"),
            "nuance": entry.get("nuance"),
            "grade": entry.get("grade"),
            "has_images": entry.get("has_images"),
            "entity_type": entry.get("entity_type"),
            "entity_id": entry.get("entity_id"),
            "reference": entry.get("reference"),
            "reference_raw": entry.get("reference_raw"),
        }
        # Clean None values
        note_json = {k: v for k, v in note_json.items() if v is not None}

        # Insert notes
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

def extract_images_from_slide(slide, slide_number: int, output_dir: Path, ppt_name: str) -> List[str]:
    """Extract all images from a slide and save them."""
    image_paths = []
    img_count = 0
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for shape in slide.shapes:
        if hasattr(shape, "image"):
            try:
                image = shape.image
                image_bytes = image.blob
                filename = f"{ppt_name}_slide{slide_number:03d}_img{img_count:02d}.png"
                filepath = images_dir / filename
                
                img = Image.open(io.BytesIO(image_bytes))
                img.save(filepath, "PNG")
                
                # Relative path for database
                rel_path = str(filepath.relative_to(output_dir.parent))
                image_paths.append(rel_path)
                img_count += 1
            except Exception as e:
                print(f"      ⚠️  Error extracting image from slide {slide_number}: {e}")
    return image_paths

def process_powerpoint(ppt_path: Path, output_dir: Path, file_id: int, force_reprocess: bool = False) -> List[Dict]:
    """Process a PowerPoint file: extract images, metadata, and insert into PostgreSQL."""
    print(f"\n📊 Processing: {ppt_path.name}")

    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)  # Register pgvector adapter
    cur = conn.cursor()

    try:
        source_file_id = file_id

        # Load presentation
        prs = Presentation(ppt_path)
        ppt_name = ppt_path.stem

        print(f"   Found {len(prs.slides)} slides")

        # First pass: extract all metadata
        all_slide_metadata = []
        for slide_idx, slide in enumerate(prs.slides, start=1):
            metadata = extract_metadata_from_slide(slide, slide_idx)
            all_slide_metadata.append(metadata)

        # Entity inheritance (propagate reference backward)
        last_found_entity = None
        for i in range(len(all_slide_metadata) - 1, -1, -1):
            sm = all_slide_metadata[i]
            gk = group_key(sm)
            if sm.get("entity_id") and gk:
                last_found_entity = {
                    "entity_id": sm["entity_id"],
                    "entity_type": sm["entity_type"],
                    "nuance": sm.get("nuance"),
                    "product_name": sm.get("product_name"),
                    "reference": sm.get("reference"),
                    "reference_raw": sm.get("reference_raw"),
                    "grade": sm.get("grade"),
                    "mag_scale_mapping": sm.get("mag_scale_mapping", {}),
                }
            elif last_found_entity:
                sm["entity_id"] = last_found_entity["entity_id"]
                sm["entity_type"] = last_found_entity["entity_type"]
                sm["nuance"] = last_found_entity.get("nuance")
                sm["product_name"] = last_found_entity.get("product_name")
                sm["reference"] = last_found_entity.get("reference")
                sm["reference_raw"] = last_found_entity.get("reference_raw")
                sm["grade"] = last_found_entity.get("grade")
                # Inherit mag_scale_mapping from previous slide
                if not sm.get("mag_scale_mapping") and last_found_entity.get("mag_scale_mapping"):
                    sm["mag_scale_mapping"] = last_found_entity["mag_scale_mapping"]
                # Resolve scale if we have magnification and mapping
                if sm.get("magnification_value") and sm.get("mag_scale_mapping"):
                    mapped_scale = sm["mag_scale_mapping"].get(sm["magnification_value"])
                    if mapped_scale and not sm.get("scale"):
                        sm["scale"] = mapped_scale

        # Aggregate texts by group
        entity_text_map = {}
        for sm in all_slide_metadata:
            gk = group_key(sm)
            if not gk:
                continue
            chunks = []
            ft = (sm.get("full_text") or "").strip()
            if ft:
                chunks.append(ft)
            comm = (sm.get("comments") or sm.get("description") or "").strip()
            if comm and comm not in ft:
                chunks.append(comm)
            ct = composition_table_to_text(sm.get("composition_table"))
            if ct:
                chunks.append("TABLEAU DE COMPOSITION:\n" + ct)
            if chunks:
                prev = entity_text_map.get(gk, "")
                merged = (prev + "\n\n" + "\n".join(chunks)).strip() if prev else "\n".join(chunks)
                entity_text_map[gk] = merged

        # Second pass: extract images and insert
        all_metadata = []
        skipped_no_reference = 0
        
        for slide_idx, slide in enumerate(prs.slides, start=1):
            current_metadata = all_slide_metadata[slide_idx - 1]
            image_paths = extract_images_from_slide(slide, slide_idx, output_dir, ppt_name)

            if image_paths:
                # Find best comments
                best_comments = current_metadata.get("comments")
                current_gk = group_key(current_metadata)
                if not best_comments or len(best_comments) < 50:
                    for lookback in range(1, 4):
                        prev_idx = slide_idx - 1 - lookback
                        if prev_idx >= 0:
                            prev_metadata = all_slide_metadata[prev_idx]
                            if group_key(prev_metadata) == current_gk:
                                prev_comm = prev_metadata.get("comments") or ""
                                if prev_comm and len(prev_comm) > 50:
                                    best_comments = prev_comm
                                    break

                # **CRITICAL FIX: Get or create matiere by reference**
                matiere_id = get_or_create_matiere_by_reference(cur, conn, current_metadata)
                
                if not matiere_id:
                    skipped_no_reference += 1
                    ref_info = current_metadata.get("reference_raw") or "N/A"
                    print(f"   ⚠️  Slide {slide_idx}: Skipped - no reference (tried: {ref_info})")
                    continue

                # Process each image
                for img_path in image_paths:
                    # Load image for embedding
                    full_img_path = output_dir.parent / img_path
                    img = Image.open(full_img_path)
                    
                    # Compute embedding
                    embedding = compute_embedding_from_pil(img)

                    entry = current_metadata.copy()
                    entry["image_path"] = img_path
                    entry["source_file"] = ppt_path.name
                    if best_comments:
                        entry["comments"] = best_comments
                        entry["description"] = best_comments

                    gk = group_key(entry)
                    if gk and gk in entity_text_map:
                        entry["entity_full_text"] = entity_text_map[gk]
                    else:
                        entry["entity_full_text"] = ""

                    # Insert into database
                    success = insert_image_and_notes(conn, cur, entry, source_file_id, matiere_id, embedding)
                    if success:
                        all_metadata.append(entry)
                        ref_display = entry.get('reference_raw') or entry.get('reference') or 'N/A'
                        mag_display = entry.get('magnification') or 'N/A'
                        scale_display = entry.get('scale') or 'N/A'
                        print(f"   ✅ Slide {slide_idx}: {Path(img_path).name} -> matiere_id={matiere_id} | ref:{ref_display} | mag:{mag_display} | scale:{scale_display}")

        if skipped_no_reference > 0:
            print(f"   ⚠️  Skipped {skipped_no_reference} slides without valid references")
        
        print(f"✅ Processed {ppt_path.name}: {len(all_metadata)} images inserted")
        return all_metadata

    except Exception as e:
        conn.rollback()
        print(f"❌ Error processing {ppt_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        cur.close()
        conn.close()

def main():
    output_dir = BASE_DIR / "embeddings_v7"
    
    # Delete and rebuild embeddings folder if it exists
    if output_dir.exists():
        print(f"🗑️  Deleting existing embeddings folder: {output_dir}")
        shutil.rmtree(output_dir)
        print("✅ Folder deleted\n")
    
    print(f"📁 Creating embeddings folder: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    cur = conn.cursor()

    try:
        # Get all PowerPoint files from database
        cur.execute("SELECT id, file_path FROM public.powerpoint_files ORDER BY id ASC")
        powerpoint_files = cur.fetchall()
        
        total_files = len(powerpoint_files)
        print(f"\n📂 Found {total_files} PowerPoint files to process\n")

        all_metadata = []
        for idx, (file_id, file_path) in enumerate(powerpoint_files, 1):
            print(f"[{idx}/{total_files}] {Path(file_path).name}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"  ⚠️  File not found: {file_path}")
                continue
            
            ppt_path = Path(file_path)
            metadata = process_powerpoint(ppt_path, output_dir, file_id=file_id, force_reprocess=False)
            all_metadata.extend(metadata)

        # Save metadata.json
        if all_metadata:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(all_metadata, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Metadata saved: {metadata_path}")

        print(f"\n🎉 Complete! Total images processed: {len(all_metadata)}")

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()