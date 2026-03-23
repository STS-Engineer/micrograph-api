#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        EXTRACTION MICROGRAPHIES — VERSION 4 (UNIFIÉE)                       ║
║                                                                              ║
║  Supporte 3 types de fichiers PowerPoint :                                  ║
║    1. Nuances métalliques  → tables: nuances / nuance_images / nuance_expert_notes ║
║    2. Cokes comparatifs    → tables: matieres / matiere_images / matiere_expert_notes ║
║    3. Standard (graphite…) → tables: matieres / matiere_images / matiere_expert_notes ║
║                                                                              ║
║  Détection automatique du type via is_metallic_nuances_file()               ║
║                                and is_cokes_comparative_file()              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import io
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
from transformers import AutoModel, AutoImageProcessor
from pptx import Presentation
from PIL import Image

import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from pgvector import Vector

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DB_DSN = "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA"

# ──────────────────────────────────────────────────────────────────────────────
# SETUP DINOv2
# ──────────────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Loading DINOv2-large on {device}...")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
print("✅ DINOv2 model loaded\n")

BASE_DIR = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITAIRES COMMUNS
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_reference(ref: str) -> str:
    if not ref:
        return ""
    ref = re.sub(r'^(ref|référence)\s*[:\-]?\s*', '', ref, flags=re.IGNORECASE)
    ref = re.sub(r'\s+', '', ref)
    return ref.strip().upper()


def compute_embedding_from_pil(image: Image.Image) -> np.ndarray:
    """Calcule l'embedding DINOv2 (1024 dimensions)."""
    image = image.convert("RGB")
    inputs = dinov2_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dinov2_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding.astype("float32")


def extract_magnifications_with_positions(slide) -> List[Dict]:
    """Extrait tous les grossissements et leur position verticale."""
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


def assign_magnification(shape_top: int, magnifications: List[Dict]) -> Optional[int]:
    """Retourne le grossissement le plus proche au-dessus d'une image."""
    if not magnifications:
        return None
    mags_above = [m for m in magnifications if m["top"] <= shape_top]
    if mags_above:
        return mags_above[-1]["value"]
    return magnifications[0]["value"]


def extract_tables_from_slide(slide) -> List[List[List[str]]]:
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


def _extract_nuance_from_text(text: str) -> Optional[str]:
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
    if not text:
        return None
    comments_match = re.search(
        r"Commentaires?\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL
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


def parse_avo_composition_from_tables(tables: List[List[List[str]]]) -> Optional[Dict[str, Any]]:
    """Extrait un tableau de composition (format AVO) depuis les tables PPT."""
    for t in tables:
        rows = [[(c or "").strip() for c in r] for r in t if any((c or "").strip() for c in r)]
        if len(rows) < 2:
            continue
        for idx in range(1, len(rows)):
            value_row = rows[idx]
            numeric_cells = [c for c in value_row if c and re.match(r"^[<>]?\s*\d+(?:[.,]\d+)?\s*$", c)]
            if len(numeric_cells) < 2:
                continue
            header_row = rows[0]
            pairs = []
            for j in range(min(len(header_row), len(value_row))):
                h = header_row[j]
                v = value_row[j]
                if h and v and re.match(r"^[<>]?\s*\d+(?:[.,]\d+)?\s*$", v):
                    pairs.append((h, v))
            if len(pairs) >= 2:
                return {
                    "elements": [h for h, _ in pairs],
                    "values": [v for _, v in pairs],
                    "rows": [[h for h, _ in pairs], [v for _, v in pairs]]
                }
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DÉTECTION DU TYPE DE FICHIER
# ══════════════════════════════════════════════════════════════════════════════

def is_metallic_nuances_file(prs: Presentation) -> bool:
    """
    Détecte les fichiers "nuances métalliques".
    Critères : slide 1 contient "nuance" + "métallique",
               slide 2 contient "Nuance" + "Commentaires" sans images.
    """
    slides = list(prs.slides)
    if len(slides) < 2:
        return False

    slide1_text = " ".join(
        shape.text for shape in slides[0].shapes if hasattr(shape, "text")
    ).lower()

    if "nuance" not in slide1_text and "métallique" not in slide1_text:
        return False

    slide2_text = " ".join(
        shape.text for shape in slides[1].shapes if hasattr(shape, "text")
    )
    slide2_has_images = any(hasattr(s, "image") for s in slides[1].shapes)

    return "Nuance" in slide2_text and "Commentaires" in slide2_text and not slide2_has_images


def is_cokes_comparative_file(prs: Presentation) -> bool:
    """
    Détecte les fichiers de comparaison Cokes.
    Critères : slide 1 contient "Coke",
               slide 2 contient ≥3 occurrences de "Micrographie N°".
    """
    slides = list(prs.slides)
    if len(slides) < 2:
        return False

    slide1_text = " ".join(
        shape.text for shape in slides[0].shapes if hasattr(shape, "text")
    )
    if "Coke" not in slide1_text:
        return False

    slide2_text = "\n".join(
        shape.text for shape in slides[1].shapes if hasattr(shape, "text")
    )
    return slide2_text.count("Micrographie N°") >= 3


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BASE DE DONNÉES : CIRCUIT NUANCES MÉTALLIQUES
#   Tables : nuances, nuance_images, nuance_expert_notes
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_nuance(cur, conn, code_nuance: str, reference: Optional[str]) -> Optional[int]:
    """
    Récupère ou crée une entrée dans public.nuances.
    Clé : reference (ex. "5500485") ou code_nuance (ex. "477 00").
    """
    if not code_nuance:
        return None

    # Cherche par référence si disponible, sinon par name
    if reference:
        cur.execute("SELECT id FROM public.nuances WHERE reference = %s", (reference,))
    else:
        cur.execute("SELECT id FROM public.nuances WHERE name = %s", (code_nuance,))

    row = cur.fetchone()
    if row:
        return row[0]

    # Crée la nuance
    cur.execute(
        """
        INSERT INTO public.nuances (reference, name, status, created_at)
        VALUES (%s, %s, 'active', NOW())
        RETURNING id
        """,
        (reference or code_nuance, code_nuance),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def insert_nuance_image(
    conn, cur,
    nuance_id: int,
    image_path: str,
    embedding: np.ndarray,
    code_nuance: str,
    reference: Optional[str],
    metadata: Dict,
    source_file_id: int,
) -> bool:
    """
    Insère dans nuance_images + nuance_expert_notes.
    Schema nuance_images : (id, nuance_id, image_path, embedding, nuance_name, reference, created_at)
    Schema nuance_expert_notes : (id, nuance_image_id, note_json, created_at)
    """
    try:
        embedding_vector = Vector(embedding.tolist())
        cur.execute(
            """
            INSERT INTO public.nuance_images
                (nuance_id, image_path, embedding, nuance_name, reference, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            RETURNING id
            """,
            (nuance_id, image_path, embedding_vector, code_nuance, reference),
        )
        image_id = cur.fetchone()[0]

        note_json = {
            "expert_notes":  metadata.get("comments") or "",
            "magnification": metadata.get("magnification"),
            "slide_number":  metadata.get("slide_number"),
            "source_file_id": source_file_id,
            "nuance":        code_nuance,
            "reference":     reference,
            "composition":   metadata.get("composition"),
            "annotations":   metadata.get("annotations", []),
            "type":          "Nuance métallique",
        }

        cur.execute(
            """
            INSERT INTO public.nuance_expert_notes (nuance_image_id, note_json, created_at)
            VALUES (%s, %s, NOW())
            """,
            (image_id, Json(note_json)),
        )
        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"      ❌ Erreur insertion nuance_images: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BASE DE DONNÉES : CIRCUIT MATIÈRES PREMIÈRES
#   Tables : matieres, matiere_images, matiere_expert_notes
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_matiere(cur, conn, entry: Dict) -> Optional[int]:
    """
    Récupère ou crée une entrée dans public.matieres.
    Clé : reference.
    """
    ref = entry.get("reference")
    if not ref:
        return None

    cur.execute("SELECT matiere_id FROM public.matieres WHERE reference = %s", (ref,))
    row = cur.fetchone()
    if row:
        return row[0]

    name = entry.get("product_name") or f"Matière {ref}"
    matiere_type = entry.get("type_matiere") or "Matière première"

    cur.execute(
        """
        INSERT INTO public.matieres (nom_matiere, type_matiere, reference, date_creation, date_mise_a_jour)
        VALUES (%s, %s, %s, NOW(), NOW())
        RETURNING matiere_id
        """,
        (name, matiere_type, ref),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    return new_id


def insert_matiere_image(
    conn, cur,
    matiere_id: int,
    image_path: str,
    embedding: np.ndarray,
    entry: Dict,
    source_file_id: int,
) -> bool:
    """
    Insère dans matiere_images + matiere_expert_notes.
    Schema matiere_images : (id, matiere_id, image_path, embedding, material_name, reference)
    Schema matiere_expert_notes : (id, matiere_image_id, note_json, created_at)
    """
    try:
        embedding_vector = Vector(embedding.tolist())
        cur.execute(
            """
            INSERT INTO public.matiere_images
                (matiere_id, image_path, embedding, material_name, reference)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                matiere_id,
                image_path,
                embedding_vector,
                entry.get("product_name"),
                entry.get("reference"),
            ),
        )
        image_id = cur.fetchone()[0]

        note_json = {
            "expert_notes":  entry.get("comments") or "",
            "magnification": entry.get("magnification"),
            "slide_number":  entry.get("slide_number"),
            "source_file_id": source_file_id,
            "reference":     entry.get("reference"),
            "composition":   entry.get("composition"),
            "type":          entry.get("type_matiere", "Matière première"),
        }

        cur.execute(
            """
            INSERT INTO public.matiere_expert_notes (matiere_image_id, note_json, created_at)
            VALUES (%s, %s, NOW())
            """,
            (image_id, Json(note_json)),
        )
        conn.commit()
        return True

    except Exception as e:
        conn.rollback()
        print(f"      ❌ Erreur insertion matiere_images: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROCESSEUR : NUANCES MÉTALLIQUES
# ══════════════════════════════════════════════════════════════════════════════

def _parse_metallic_composition_table(table) -> Optional[Dict[str, Any]]:
    """Extrait la composition depuis un tableau PPT (format nuances métalliques)."""
    rows = [[c.text.strip() for c in r.cells] for r in table.rows]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return None

    headers = rows[0]
    value_row = None
    for r in reversed(rows):
        if any(re.match(r"^[\d,\.]+$", c) for c in r if c):
            value_row = r
            break
    if not value_row:
        return None

    ref_row = rows[1] if len(rows) > 2 else None
    components = []
    for j, header in enumerate(headers):
        if not header:
            continue
        val = value_row[j] if j < len(value_row) else ""
        ref = ref_row[j] if ref_row and j < len(ref_row) else ""
        if header and val:
            components.append({"name": header, "reference": ref or None, "percentage": val})

    return {"components": components} if components else None


def _parse_metallic_metadata_slide(slide) -> Dict[str, Any]:
    """
    Extrait les métadonnées d'une slide "Nuance XXXX XX".
    Retourne: nuance, reference, comments, composition.
    """
    result: Dict[str, Any] = {
        "nuance": None, "reference": None, "comments": None, "composition": None
    }

    for shape in slide.shapes:
        text = shape.text.strip() if hasattr(shape, "text") else ""

        if "Nuance" in text and result["nuance"] is None:
            # Ex: "Nuance 477 00" ou "Nuance 377 A19"
            m = re.search(r"Nuance\s+([A-Z0-9]{3}\s+[A-Z0-9]{2,3})", text, re.IGNORECASE)
            if m:
                result["nuance"] = m.group(1).strip()
            # Référence interne ex: "( 5500 485 )"
            ref_m = re.search(r"\(\s*(\d{4}\s+\d{3})\s*\)", text)
            if ref_m:
                result["reference"] = ref_m.group(1).replace(" ", "")

        if "Commentaires" in text and result["comments"] is None:
            m = re.search(r"Commentaires\s*:\s*\n?(.*)", text, re.DOTALL)
            if m:
                raw = re.sub(r"[ \t]+", " ", m.group(1).strip())
                raw = re.sub(r"\n{3,}", "\n\n", raw)
                if len(raw) > 10:
                    result["comments"] = raw

        if getattr(shape, "has_table", False) and result["composition"] is None:
            result["composition"] = _parse_metallic_composition_table(shape.table)

    return result


def _group_slides_by_nuance(slides: list) -> List[Dict]:
    """
    Regroupe les slides en blocs [metadata_slide + image_slides].
    Un header de nuance = slide contenant "Nuance" + "Commentaires" sans images.
    """
    groups = []
    current_group = None

    for i, slide in enumerate(slides):
        slide_text = " ".join(
            shape.text for shape in slide.shapes if hasattr(shape, "text")
        )
        has_images = any(hasattr(s, "image") for s in slide.shapes)
        is_metadata = (
            "Nuance" in slide_text
            and "Commentaires" in slide_text
            and not has_images
        )

        if is_metadata:
            current_group = {
                "metadata_slide": slide,
                "image_slides": [],
                "slide_indices": {"metadata": i + 1, "images": []},
            }
            groups.append(current_group)
        elif current_group is not None and has_images:
            current_group["image_slides"].append(slide)
            current_group["slide_indices"]["images"].append(i + 1)

    return groups


def _extract_annotations_from_image_slide(slide) -> List[str]:
    """Extrait les labels courts sur une slide d'images (ex: 'Cuivre', 'MoS2')."""
    annotations = []
    for shape in slide.shapes:
        if hasattr(shape, "text") and shape.text.strip():
            text = shape.text.strip()
            if (
                len(text) < 60
                and "Grossissement" not in text
                and not re.match(r"^\d+$", text)
                and not re.match(r"^\d{3}\s+[A-Z0-9]{2,3}$", text)
                and "Commentaires" not in text
            ):
                annotations.append(clean_text(text))
    return annotations


def process_metallic_powerpoint(ppt_path: Path, output_dir: Path, file_id: int):
    """
    Traite un fichier PowerPoint de nuances métalliques.
    Insère dans: nuances / nuance_images / nuance_expert_notes
    """
    print(f"\n🔬 Traitement [NUANCES MÉTALLIQUES] : {ppt_path.name}")

    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    cur = conn.cursor()

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        prs = Presentation(ppt_path)
        slides = list(prs.slides)
        groups = _group_slides_by_nuance(slides)
        print(f"   → {len(groups)} nuances détectées")

        total_images = 0

        for group in groups:
            meta = _parse_metallic_metadata_slide(group["metadata_slide"])
            code_nuance = meta["nuance"]
            reference   = meta["reference"]
            comments    = meta["comments"]
            composition = meta["composition"]

            if not code_nuance:
                print(f"   ⚠️  Nuance non détectée dans la slide de métadonnées, ignorée")
                continue

            nuance_id = get_or_create_nuance(cur, conn, code_nuance, reference)
            if not nuance_id:
                print(f"   ⚠️  Impossible de créer la nuance {code_nuance}")
                continue

            ref_str = f"ref {reference}" if reference else "sans ref"
            print(f"\n   📌 Nuance {code_nuance} ({ref_str})")

            img_global = 0
            for slide_obj, slide_num in zip(
                group["image_slides"], group["slide_indices"]["images"]
            ):
                magnifications = extract_magnifications_with_positions(slide_obj)
                annotations    = _extract_annotations_from_image_slide(slide_obj)

                # Commentaires inline sur la slide d'images (complément)
                inline_comments = None
                for shape in slide_obj.shapes:
                    if hasattr(shape, "text") and "Commentaires" in shape.text:
                        m = re.search(r"Commentaires\s*:\s*\n?(.*)", shape.text, re.DOTALL)
                        if m:
                            raw = re.sub(r"[ \t]+", " ", m.group(1).strip())
                            if len(raw) > 10:
                                inline_comments = raw
                                break

                effective_comments = comments
                if inline_comments and not comments:
                    effective_comments = inline_comments
                elif inline_comments and comments:
                    effective_comments = comments + "\n\n[Annotations]\n" + inline_comments

                for shape in slide_obj.shapes:
                    if not hasattr(shape, "image"):
                        continue

                    mag = assign_magnification(shape.top, magnifications)
                    img_bytes = shape.image.blob
                    filename  = f"{ppt_path.stem}_s{slide_num:03d}_i{img_global:02d}.png"
                    filepath  = images_dir / filename

                    img = Image.open(io.BytesIO(img_bytes))
                    img.save(filepath, "PNG")
                    embedding = compute_embedding_from_pil(img)

                    try:
                        rel_path = str(filepath.relative_to(output_dir.parent))
                    except ValueError:
                        rel_path = str(filepath)

                    success = insert_nuance_image(
                        conn, cur,
                        nuance_id=nuance_id,
                        image_path=rel_path,
                        embedding=embedding,
                        code_nuance=code_nuance,
                        reference=reference,
                        metadata={
                            "comments":      effective_comments,
                            "composition":   composition,
                            "magnification": mag,
                            "slide_number":  slide_num,
                            "annotations":   annotations,
                        },
                        source_file_id=file_id,
                    )

                    if success:
                        mag_str = f"x{mag}" if mag else "?"
                        com_str = f"✓ ({len(effective_comments)} chars)" if effective_comments else "✗"
                        print(f"      ✅ Slide {slide_num} img {img_global:02d} | Gross. {mag_str} | Com. {com_str}")
                        total_images += 1
                    img_global += 1

        print(f"\n   ✅ Terminé : {total_images} image(s) insérée(s) [nuance_images]")

    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback; traceback.print_exc()
    finally:
        cur.close()
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PROCESSEUR : COKES COMPARATIFS
# ══════════════════════════════════════════════════════════════════════════════

def extract_cokes_references_dict(prs: Presentation) -> Dict[str, str]:
    """Extrait les références depuis le Slide 1 (page de titre Cokes)."""
    if not prs.slides:
        return {}
    slide = list(prs.slides)[0]
    text_blocks = [shape.text.strip() for shape in slide.shapes
                   if hasattr(shape, "text") and shape.text.strip()]
    full_text = "\n".join(text_blocks)

    parsed = []
    for line in full_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^(.+?)\s+ref\s+(\d{5,7})\s*$', line, re.IGNORECASE)
        if m:
            product = m.group(1).strip()
            if not re.match(r'^coke\b', product, re.IGNORECASE):
                product = 'Coke ' + product
            parsed.append((product, m.group(2).strip()))
            continue
        m = re.match(r'^(.+?)\s+\(\s*(V\d+)\s*\)\s*$', line)
        if m:
            product = m.group(1).strip()
            if not re.match(r'^coke\b', product, re.IGNORECASE):
                product = 'Coke ' + product
            parsed.append((product, m.group(2).strip()))
            continue
        if re.search(r'(Micrographies|R&D|Date|\d{2}/\d{2}/\d{4})', line, re.IGNORECASE):
            continue
        parsed.append((line, None))

    ref_dict = {}
    for i, (product, ref) in enumerate(parsed):
        if ref is None:
            lookahead_ref = next((parsed[j][1] for j in range(i + 1, len(parsed)) if parsed[j][1]), None)
            if lookahead_ref:
                if not re.match(r'^coke\b', product, re.IGNORECASE):
                    product = 'Coke ' + product
                ref_dict[product] = lookahead_ref
        else:
            ref_dict[product] = ref
    return ref_dict


def extract_cokes_comments_dict(prs: Presentation) -> Dict[str, str]:
    """Extrait les commentaires depuis le Slide 2 (Cokes)."""
    slides = list(prs.slides)
    if len(slides) < 2:
        return {}
    slide = slides[1]
    text_blocks = [shape.text.strip() for shape in slide.shapes
                   if hasattr(shape, "text") and shape.text.strip()]
    full_text = "\n".join(text_blocks)

    products = [
        "Coke MUCO Cyclam", "Coke FC 250", "Coke PDS 1183",
        "Coke CBH LPCS60", "Coke CBH LPCS 60", "Coke micronisé",
        "Coke MUCO 0-75µm", "Coke MUCO 0-75 µm",
        "Coke CARBOLEG FCB 97", "Coke CARBOLEG FCB97",
    ]
    comments_dict = {}
    for product in products:
        pat = re.escape(product).replace(r"\ ", r"\s*")
        pattern = rf"{pat}\s*(?:\([^)]+\))?\s*(?:–\s*Ref\s+\d+\s+\d+\s*)?\s*:\s*\tMicrographie N° \d+\s*\n(.+?)(?=\n\n[A-Z]|\Z)"
        match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
        if match:
            raw = re.sub(r'[ \t]+', ' ', match.group(1).strip())
            raw = re.sub(r'\n{3,}', '\n\n', raw).strip()
            comments_dict[re.sub(r'\s+', ' ', product)] = raw
    return comments_dict


def _match_cokes_product(product_name: str, lookup_dict: Dict[str, str]) -> Optional[str]:
    """Match un nom de produit (insensible aux espaces et suffixes)."""
    product_name = re.sub(r'\s+', ' ', product_name).strip()
    if product_name in lookup_dict:
        return lookup_dict[product_name]
    product_base = re.sub(r'\s*\([^)]+\)', '', product_name).strip()
    for key, val in lookup_dict.items():
        key_base = re.sub(r'\s*\([^)]+\)', '', key).strip()
        prod_norm = re.sub(r'\s+', '', product_base).lower()
        key_norm  = re.sub(r'\s+', '', key_base).lower()
        if prod_norm == key_norm or (key_norm and (key_norm in prod_norm or prod_norm in key_norm)):
            return val
    return None


def process_cokes_powerpoint(ppt_path: Path, output_dir: Path, file_id: int):
    """
    Traite un fichier PowerPoint de comparaison Cokes.
    Insère dans: matieres / matiere_images / matiere_expert_notes
    """
    print(f"\n🪨 Traitement [COKES COMPARATIFS] : {ppt_path.name}")

    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    cur = conn.cursor()

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        prs = Presentation(ppt_path)
        slides = list(prs.slides)

        print("   Phase 1: Références...")
        ref_dict = extract_cokes_references_dict(prs)
        print(f"   → {len(ref_dict)} références : {list(ref_dict.items())}")

        print("   Phase 2: Commentaires...")
        comments_dict = extract_cokes_comments_dict(prs)
        print(f"   → {len(comments_dict)} produits avec commentaires")

        print("   Phase 3: Images...")
        total_images = 0

        for i in range(2, len(slides)):
            slide = slides[i]
            has_images = any(hasattr(shape, "image") for shape in slide.shapes)
            if not has_images:
                continue

            # Nom du produit sur la slide
            product_name = None
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    text = shape.text.strip()
                    if "Coke" in text and "x " not in text.lower() and "X " not in text:
                        product_name = text
                        break

            if not product_name:
                print(f"   ⚠️  Slide {i+1}: Pas de nom de produit")
                continue

            comments  = _match_cokes_product(product_name, comments_dict)
            reference = _match_cokes_product(product_name, ref_dict)

            matiere_id = get_or_create_matiere(cur, conn, {
                "reference":    reference,
                "product_name": product_name,
                "type_matiere": "Coke",
            })
            if not matiere_id:
                print(f"   ⚠️  Slide {i+1}: Impossible de créer la matière pour '{product_name}'")
                continue

            magnifications = extract_magnifications_with_positions(slide)
            img_count = 0

            for shape in slide.shapes:
                if not hasattr(shape, "image"):
                    continue

                mag = assign_magnification(shape.top, magnifications)
                img_bytes = shape.image.blob
                filename  = f"{ppt_path.stem}_s{i+1:03d}_i{img_count:02d}.png"
                filepath  = images_dir / filename

                img = Image.open(io.BytesIO(img_bytes))
                img.save(filepath, "PNG")
                embedding = compute_embedding_from_pil(img)

                try:
                    rel_path = str(filepath.relative_to(output_dir.parent))
                except ValueError:
                    rel_path = str(filepath)

                success = insert_matiere_image(
                    conn, cur,
                    matiere_id=matiere_id,
                    image_path=rel_path,
                    embedding=embedding,
                    entry={
                        "product_name": product_name,
                        "reference":    reference,
                        "comments":     comments,
                        "magnification": mag,
                        "slide_number": i + 1,
                        "composition":  {},
                        "type_matiere": "Coke",
                    },
                    source_file_id=file_id,
                )

                if success:
                    ref_str = f"Ref: {reference}" if reference else "Sans ref"
                    com_str = f"✓ ({len(comments)} chars)" if comments else "✗"
                    print(f"   ✅ Slide {i+1}: {product_name:35s} | {ref_str:15s} | Com. {com_str}")
                    total_images += 1
                img_count += 1

        print(f"\n   ✅ Terminé : {total_images} image(s) insérée(s) [matiere_images]")

    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback; traceback.print_exc()
    finally:
        cur.close()
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PROCESSEUR : STANDARD (graphite, matières premières…)
# ══════════════════════════════════════════════════════════════════════════════

def extract_metadata_from_slide(slide, slide_number: int) -> Dict:
    """Extrait les métadonnées d'une slide standard."""
    metadata = {
        "slide_number": slide_number,
        "nuance": None,
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
                    # Pattern 1 : "ref XXXXX"
                    ref_match = re.search(
                        r'(?:ref|référence)\s*[:\-]?\s*([A-Z0-9\s]{4,15})',
                        text, re.IGNORECASE
                    )
                    if ref_match:
                        metadata["reference_raw"] = ref_match.group(1).strip()
                        metadata["reference"] = clean_reference(metadata["reference_raw"])
                    # Pattern 2 : "– RSxxx"
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

    if slide_number == 1 or (
        metadata["full_text"].count("ref") > 3
        and "Commentaires" not in metadata["full_text"]
    ):
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


def process_standard_powerpoint(ppt_path: Path, output_dir: Path, file_id: int):
    """
    Traite un fichier PowerPoint standard (graphite, matières premières…).
    Insère dans: matieres / matiere_images / matiere_expert_notes
    """
    print(f"\n📊 Traitement [STANDARD] : {ppt_path.name}")

    conn = psycopg2.connect(DB_DSN)
    register_vector(conn)
    cur = conn.cursor()

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        prs = Presentation(ppt_path)
        slides = list(prs.slides)
        all_meta = []

        # Passe 1 : extraction des métadonnées
        for i, slide in enumerate(slides, 1):
            meta = extract_metadata_from_slide(slide, i)
            all_meta.append(meta)
            ref_str = f"Ref: {meta['reference']}" if meta["reference"] else "Aucune ref"
            com_str = f"✓ ({len(meta['comments'])} chars)" if meta["comments"] else "✗"
            print(f"   Slide {i}: {ref_str} | Commentaires {com_str}")

        # Passe 2 : héritage des métadonnées entre slides
        for i in range(len(all_meta)):
            current = all_meta[i]
            if not current["has_images"]:
                continue

            if not current["reference"] and i > 0:
                prev = all_meta[i - 1]
                current["reference"]     = prev["reference"]
                current["reference_raw"] = prev.get("reference_raw")
                current["nuance"]        = prev["nuance"]
                if not current["comments"] and prev["comments"]:
                    current["comments"] = prev["comments"]
                    print(f"   🔄 Slide {i+1}: métadonnées héritées de la slide {i}")
                if not current["composition"] and prev["composition"]:
                    current["composition"] = prev["composition"]

            elif current["reference"] and not current["comments"]:
                for j in range(i - 1, -1, -1):
                    prev = all_meta[j]
                    if prev["reference"] == current["reference"] and prev["comments"]:
                        current["comments"]    = prev["comments"]
                        current["nuance"]      = current["nuance"] or prev["nuance"]
                        current["composition"] = current["composition"] or prev["composition"]
                        print(f"   🔄 Slide {i+1}: commentaires hérités de la slide {j+1}")
                        break

        # Passe 3 : insertion des images
        total_images = 0
        for i, slide in enumerate(slides):
            meta = all_meta[i]
            if not meta["has_images"]:
                continue

            matiere_id = get_or_create_matiere(cur, conn, meta)
            if not matiere_id:
                print(f"   ⚠️  Slide {i+1}: Pas de référence, images ignorées")
                continue

            img_count = 0
            for shape in slide.shapes:
                if not hasattr(shape, "image"):
                    continue

                mag = assign_magnification(shape.top, meta["magnifications"])
                img_bytes = shape.image.blob
                filename  = f"{ppt_path.stem}_s{i+1:03d}_i{img_count:02d}.png"
                filepath  = images_dir / filename

                img = Image.open(io.BytesIO(img_bytes))
                img.save(filepath, "PNG")
                embedding = compute_embedding_from_pil(img)

                try:
                    rel_path = str(filepath.relative_to(output_dir.parent))
                except ValueError:
                    rel_path = str(filepath)

                success = insert_matiere_image(
                    conn, cur,
                    matiere_id=matiere_id,
                    image_path=rel_path,
                    embedding=embedding,
                    entry={
                        "product_name": meta.get("product_name"),
                        "reference":    meta["reference"],
                        "comments":     meta["comments"],
                        "magnification": mag,
                        "slide_number": i + 1,
                        "composition":  meta["composition"],
                        "type_matiere": "Matière première",
                    },
                    source_file_id=file_id,
                )

                if success:
                    com_str = (
                        f"✓ '{meta['comments'][:50]}...'" if meta["comments"] and len(meta["comments"]) > 50
                        else f"✓ '{meta['comments']}'" if meta["comments"]
                        else "✗"
                    )
                    print(f"   ✅ Slide {i+1} img {img_count} | Gross. x{mag} | Com. {com_str}")
                    total_images += 1
                img_count += 1

        print(f"\n   ✅ Terminé : {total_images} image(s) insérée(s) [matiere_images]")

    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback; traceback.print_exc()
    finally:
        cur.close()
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ROUTER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def process_powerpoint(ppt_path: Path, output_dir: Path, file_id: int):
    """
    Point d'entrée unique.
    Détecte automatiquement le type de fichier et appelle le bon processeur.

    ┌─────────────────────────────┬──────────────────────────────────────────┐
    │ Type détecté                │ Tables cibles                            │
    ├─────────────────────────────┼──────────────────────────────────────────┤
    │ Nuances métalliques         │ nuances / nuance_images /                │
    │                             │ nuance_expert_notes                      │
    ├─────────────────────────────┼──────────────────────────────────────────┤
    │ Cokes comparatifs           │ matieres / matiere_images /              │
    │                             │ matiere_expert_notes                     │
    ├─────────────────────────────┼──────────────────────────────────────────┤
    │ Standard (graphite…)        │ matieres / matiere_images /              │
    │                             │ matiere_expert_notes                     │
    └─────────────────────────────┴──────────────────────────────────────────┘
    """
    try:
        prs = Presentation(ppt_path)

        if is_metallic_nuances_file(prs):
            process_metallic_powerpoint(ppt_path, output_dir, file_id)
        elif is_cokes_comparative_file(prs):
            process_cokes_powerpoint(ppt_path, output_dir, file_id)
        else:
            process_standard_powerpoint(ppt_path, output_dir, file_id)

    except Exception as e:
        print(f"❌ Erreur détection type {ppt_path.name}: {e}")
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — NETTOYAGE
# ══════════════════════════════════════════════════════════════════════════════

def clear_all_data(output_dir: Path):
    """
    Supprime toutes les données des deux circuits (matières + nuances).
    À utiliser avec précaution avant un retraitement complet.
    """
    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        print("🗑️  Nettoyage des données existantes...")

        # Circuit matières premières
        cur.execute("DELETE FROM public.matiere_expert_notes")
        cur.execute("DELETE FROM public.matiere_images")
        print("   ✅ matiere_expert_notes / matiere_images vidées")

        # Circuit nuances métalliques
        cur.execute("DELETE FROM public.nuance_expert_notes")
        cur.execute("DELETE FROM public.nuance_images")
        print("   ✅ nuance_expert_notes / nuance_images vidées")

        conn.commit()
        cur.close()
        conn.close()

        # Fichiers images
        images_dir = output_dir / "images"
        if images_dir.exists():
            try:
                shutil.rmtree(images_dir)
                print("   ✅ Fichiers images supprimés")
            except PermissionError:
                print("   ⚠️  Dossier images verrouillé, continuation...")

        print("✅ Nettoyage terminé\n")

    except Exception as e:
        print(f"❌ Erreur nettoyage : {e}")
        import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    output = BASE_DIR / "output_v4"
    output.mkdir(exist_ok=True)

    clear_all_data(output)

    try:
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute("SELECT id, file_path FROM public.powerpoint_files ORDER BY id")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if rows:
            print(f"📂 {len(rows)} fichier(s) trouvé(s) en base\n")
            for row in rows:
                file_id  = row[0]
                ppt_path = Path(row[1])
                if ppt_path.exists():
                    process_powerpoint(ppt_path, output, file_id)
                else:
                    print(f"⚠️  Fichier introuvable : {ppt_path}")
        else:
            print("⚠️  Aucun fichier PowerPoint en base de données")

    except Exception as e:
        print(f"❌ Erreur lecture base : {e}")
        import traceback; traceback.print_exc()