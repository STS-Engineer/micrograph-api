# Micrographie IA — Materials Intelligence Platform

A full-stack **Materials Intelligence Platform** for carbon and graphite engineering, combining computer vision (DINOv2), LLM-powered analysis (Groq/OpenAI), and vector similarity search (pgvector) to identify, classify, and document materials from micrograph images.

---

## Architecture Overview

The platform runs as **two distinct Flask services**:

| Service | Host | File | Role |
|---|---|---|---|
| **Main API** | `micrographie-ia.azurewebsites.net` | `app.py` | Material search, Black Mix/Nuance management, ADN generation, DOCX export |
| **OCR Service** | `paddle-ocr.azurewebsites.net` | `ocr_api.py` | Document processing (PDF/image), specification extraction via PaddleOCR |

**Tech Stack**: Python 3 / Flask, PostgreSQL + pgvector, DINOv2 (vision embeddings), Groq Llama 3.3 70B, OpenAI GPT-4o, Azure Blob Storage, Vanilla JS frontend.

---

## Table of Contents

- [Features](#features)
- [Assistants](#assistants)
  - [1. Matières Premières](#1️⃣-matières-premières)
  - [2. Black Mix](#2️⃣-black-mix)
  - [3. Nuances Métalliques](#3️⃣-nuances-métalliques)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
  - [Main API — app.py](#main-api--apppy)
  - [OCR Service — ocr_api.py](#ocr-service--ocr_apipy)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Database Schema](#database-schema)
- [Migration Scripts](#migration-scripts)
- [Knowledge Bases](#knowledge-bases)
- [Frontend UI](#frontend-ui)
- [Key Design Patterns](#key-design-patterns)

---

## Assistants

The platform exposes three domain-specific assistants, each with dedicated missions and API endpoints.

---

### 1️⃣ Matières Premières
*Documents, ADN, opportunités, micrographie*

| # | Mission | Description |
|---|---|---|
| MP-1 | **OCR → Save** | Extract specs from PDF/XLS (datasheet, MSDS, lab_control) → save to DB |
| MP-2 | **ADN + DOCX** | Retrieve full material ADN fiche + generate DOCX report |
| MP-3 | **Opportunities** | Commercial analysis for AVOCarbon application domains |
| MP-4 | **Micrographic Analysis** | Upload image → DINOv2 similarity search → structured comparison report |

| Method | Endpoint | operationId | Mission |
|---|---|---|---|
| `POST` | `/process-pdf-to-ocr` *(OCR service)* | `processPdfToOcr` | MP-1 |
| `POST` | `/convert_xls_to_json` | `convertXlsToJson` | MP-1 |
| `POST` | `/save-ocr-result` *(OCR service)* | `saveOcrResult` | MP-1 |
| `PUT` | `/update-specification-by-reference` *(OCR service)* | `updateSpecificationByReference` | MP-1 |
| `GET` | `/fiche_adn?reference=` | `getFicheAdnByReference` | MP-2 |
| `GET` | `/fiche_adn/{matiere_id}` | `getFicheAdnById` | MP-2 |
| `GET` | `/material_details/{matiere_id}` | `getMaterialDetails` | MP-2 |
| `GET` | `/generate_fiche_adn_docx?reference=` | `generateFicheAdnDocx` | MP-2 |
| `POST` | `/generate_application_analysis` | `generateApplicationAnalysis` | MP-3 |
| `GET` | `/application_analysis?reference=` | `getApplicationAnalysis` | MP-3 |
| `POST` | `/upload_and_search` | `uploadAndSearch` | MP-4 |
| `POST` | `/search` | `searchSimilarMaterials` | MP-4 |

---

### 2️⃣ Black Mix
*Formulation, ADN, DOCX — pas d'analyse image*

| # | Mission | Description |
|---|---|---|
| BM-1 | **Submit** | Parse Mischkarte (XLS/PDF) → validate all refs → submit formulation → auto-generate ADN |
| BM-2 | **Update** | Full replacement of an existing Black Mix (composition, steps, control plan, ADN refresh) |
| BM-3 | **ADN** | Retrieve ADN at 3 levels: `basic` (stored snapshot), `combined` (+ light summaries), `enriched` (+ live AI) |
| BM-4 | **DOCX** | Generate professional ADN report via Groq LLM |

| Method | Endpoint | operationId | Mission |
|---|---|---|---|
| `POST` | `/convert_xls_to_json` | `convertXlsToJson` | BM-1 |
| `GET` | `/black-mix/validate-material/{reference}` | `validateBlackMixMaterial` | BM-1 |
| `POST` | `/black-mix/submit` | `submitBlackMixData` | BM-1 |
| `PUT` | `/black-mix/{mix_id}/update` | `updateBlackMix` | BM-2 |
| `GET` | `/black-mix/list` | `listBlackMixes` | BM-3 |
| `GET` | `/black-mix/{mix_id}` | `getBlackMixDetails` | BM-3 |
| `GET` | `/black-mix/{mix_id}/adn?level=` | `getBlackMixAdn` | BM-3 |
| `GET` | `/generate_black_mix_adn_docx?mix_id=` | `generateBlackMixAdnDocx` | BM-4 |

---

### 3️⃣ Nuances Métalliques
*Nuances, cuisson, ADN, DOCX, similarité image*

| # | Mission | Description |
|---|---|---|
| NU-1 | **Submit** | Parse Mischkarte + cuisson (Wärme-Nachbehandlung) → validate all refs → submit → auto-generate ADN |
| NU-2 | **Update** | Full replacement of an existing Nuance (composition, steps, cuisson, control plan, ADN refresh) |
| NU-3 | **ADN** | Retrieve ADN recursively (nuance + all components: matière / black_mix / nuance) |
| NU-4 | **DOCX** | Generate professional ADN report via Groq LLM |
| NU-5 | **Image Similarity** | Upload image → DINOv2 similarity search against nuance micrographs |

| Method | Endpoint | operationId | Mission |
|---|---|---|---|
| `POST` | `/convert_xls_to_json` | `convertXlsToJson` | NU-1 |
| `GET` | `/nuance/validate-material/{reference}` | `validateNuanceMaterial` | NU-1 |
| `POST` | `/nuance/submit` | `submitNuance` | NU-1 |
| `PUT` | `/nuance/{nuance_id}/update` | `updateNuance` | NU-2 |
| `GET` | `/nuance/list` | `listNuances` | NU-3 |
| `GET` | `/nuance/{nuance_id}` | `getNuanceDetails` | NU-3 |
| `GET` | `/nuance/{nuance_id}/adn?level=` | `getNuanceAdn` | NU-3 |
| `GET` | `/generate_nuance_adn_docx?nuance_id=` | `generateNuanceAdnDocx` | NU-4 |
| `POST` | `/nuance/search-similar` | `searchSimilarNuances` | NU-5 |
| `GET` | `/cuisson-programs/{program_number}` | `getCuissonProgramDetail` | NU-1/2 |
| `GET` | `/cuisson-programs/parse?value=` | `parseCuissonField` | NU-1/2 |

---

## Features

### Image-Based Material Search
- Upload a micrograph image → multi-scale DINOv2 embedding (4 crop views: 100%, 92%, 80%, 66%) → cosine similarity search via pgvector → return top-K matching materials with confidence scores.

### Two-Image Comparison
- Upload two micrographs side-by-side → independent search for each → cross-embedding cosine similarity → Groq LLM generates a structured differential analysis (Differences / ADN-A / ADN-B).

### Black Mix Formulation
- Full CRUD for Black Mix recipes: composition (components + percentages), process steps (Arbeitsfolge), step-material mapping, control plans.
- Auto-generates an **ADN snapshot** (compact material identity) on each submission.
- DOCX report export via Groq LLM.

### Nuances Métalliques
- Full CRUD for metallic shade formulations with baking program (Wärme-Nachbehandlung) support.
- Cuisson program parsing (e.g., `"101 25"` → program 101, 25% H₂).
- Recursive ADN retrieval walks through component references.

### Raw Materials (Matières Premières)
- OCR extraction from PDF/image → structured specifications (datasheet, MSDS, lab_control).
- ADN consultation with full specification aggregation.
- Application/opportunity analysis for AVOCarbon domains.

### Document Processing (OCR Service)
- PaddleOCR with automatic orientation correction.
- PDF processing via PyMuPDF with rotation detection.
- Smart component resolution: determine if a reference points to a matière, Black Mix, or nuance.
- Excel parsing: auto-detect Mischkarte (composition) vs. VM-xxx (control plan) formats.

---

## Project Structure

```
micrograph-api/
├── .env                                # Environment variables (not committed — see .gitignore)
├── .gitignore                          # Git ignore rules
├── app.py                              # Main Flask API (material search, CRUD, ADN, DOCX)
├── ocr_api.py                          # OCR Service (PaddleOCR, PDF, spec extraction)
├── openapi.yaml                        # OpenAPI spec — Main API endpoints
├── openapi2.yaml                       # OpenAPI spec — OCR Service endpoints
├── requirements.txt                    # Python dependencies
├── system_prompt.txt                   # GPT Actions system prompt (router rules)
├── startup.sh                          # Gunicorn startup script
│
├── black_mix_kb.md                     # Knowledge base — Black Mix workflows
├── matieres_premieres_kb.md            # Knowledge base — Raw Materials workflows
├── nuances_metalliques_kb.md           # Knowledge base — Nuances workflows
│
├── populate_database_v2.py             # Batch PPT image import + DINOv2 embeddings
├── migrate_image_paths_to_azure.py     # Local paths → Azure Blob URI migration
├── migrate_normalize_embeddings.py     # L2-normalize stored embedding vectors
├── upload_images_to_azure.py           # Bulk upload images to Azure Blob Storage
├── upload_folders_to_azure.py          # Bulk upload arbitrary folders to Azure Blob Storage
│
├── static/
│   ├── materials_intelligence_ui.css   # Frontend styles (design tokens + components)
│   └── materials_intelligence_ui.js    # Frontend logic (dual-view orchestration)
├── templates/
│   └── materials_intelligence_ui.html  # Jinja2 template (search + compare views)
│
├── embeddings_v7/images/               # Local image cache (populated at runtime)
├── temp_uploads/                       # Temporary upload staging
└── temp_docx/                          # Temporary DOCX export files
```

---

## API Reference

### Main API — `app.py`

**Base URL**: `https://micrographie-ia.azurewebsites.net`

#### Image Search

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload_and_search` | Upload micrograph → DINOv2 embedding → pgvector kNN search |
| `POST` | `/search` | Alternative image search endpoint |
| `POST` | `/compare_micrographs` | Two-image comparison with differential ADN analysis |

#### Matières Premières (Raw Materials)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/matieres` | List all raw materials |
| `POST` | `/save-matiere-data` | Create/update a raw material |
| `POST` | `/generate_fiche_adn_docx` | Generate ADN DOCX report for a material |
| `GET` | `/generate_fiche_adn_docx` | Download generated DOCX file |
| `POST` | `/generate_application_analysis` | Generate opportunity analysis |
| `GET` | `/application_analysis` | Retrieve application analysis |

#### Black Mix

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/black-mix/submit` | Submit a new Black Mix formulation |
| `PUT` | `/black-mix/{id}/update` | Full replacement update |
| `GET` | `/black-mix/validate-material/{ref}` | Validate a material reference |
| `GET` | `/black-mix/{id}/adn` | Retrieve ADN (`?level=basic\|combined\|enriched`) |
| `POST` | `/generate_black_mix_adn_docx` | Generate DOCX report |
| `GET` | `/black-mix/list` | List all Black Mixes |

#### Nuances Métalliques

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/nuance/submit` | Submit a new nuance formulation |
| `PUT` | `/nuance/{id}/update` | Full replacement update |
| `GET` | `/nuance/{id}/adn` | Retrieve ADN (`?level=basic\|combined\|enriched`) |
| `POST` | `/generate_nuance_adn_docx` | Generate DOCX report |
| `GET` | `/nuance/list` | List all nuances |

#### Cuisson Programs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/cuisson-programs/{program_number}` | Get one baking program by number |
| `GET` | `/cuisson-programs/parse?value=101 25` | Parse a raw cuisson field and return interpreted values |

#### Utilities

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/convert_xls_to_json` | Parse Excel → JSON (auto-detect Mischkarte vs. VM-xxx) |
| `GET` | `/` | Serve the Materials Intelligence UI |

---

### OCR Service — `ocr_api.py`

**Base URL**: `https://paddle-ocr.azurewebsites.net`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/process-pdf-to-ocr` | Upload PDF/image → PaddleOCR text extraction |
| `POST` | `/save-ocr-result` | Persist extracted specification data |
| `PUT` | `/update-specification-by-reference` | Update specification for a reference |
| `POST` | `/black-mix/submit` | Submit Black Mix (duplicate endpoint) |
| `GET` | `/black-mix/validate-material/{ref}` | Validate material reference |
| `POST` | `/black-mix/resolve-component` | Resolve component type (matière/black_mix/nuance) |

> **Note**: `/black-mix/submit` and `/black-mix/validate-material` exist on both services. The canonical endpoints are on the Main API.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key (GPT-4o for analysis) |
| `GROQ_API_KEY` | Yes | — | Groq API key (Llama 3.3 70B for ADN/DOCX generation) |
| `AZURE_CONNECTION_STRING` | Yes | — | Azure Blob Storage connection string |
| `AZURE_CONTAINER_NAME` | Yes | — | Blob container name |
| `AZURE_BLOB_PREFIX` | No | `micrograph-images` | Blob path prefix for images |
| `EXPECTED_AZURE_CONTAINER_NAME` | No | `micrographie-images` | Expected container name for URL validation |
| `AGENT_DEBUG_TOKEN` | No | — | Token for debug logging mode |

---

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension
- Azure Blob Storage account

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd micrograph-api

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note**: PyTorch is installed from the CPU-only index (`--extra-index-url https://download.pytorch.org/whl/cpu`). For GPU support, modify the index URL accordingly.

### Key Dependencies

| Category | Packages |
|---|---|
| **AI / Vision** | `torch`, `torchvision`, `transformers` (DINOv2) |
| **LLM Clients** | `openai`, `groq` |
| **OCR** | `paddleocr`, `paddlepaddle` |
| **Database** | `psycopg2-binary`, `pgvector` |
| **Azure** | `azure-storage-blob` |
| **Document Gen** | `python-docx`, `python-pptx` |
| **PDF Processing** | `PyMuPDF` (fitz) |
| **Excel** | `xlrd`, `openpyxl` |
| **Image** | `Pillow`, `opencv-python` |

---

## Running Locally

### Main API

```bash
# Set environment variables
export OPENAI_API_KEY=<your-key>
export GROQ_API_KEY=<your-key>
export AZURE_CONNECTION_STRING=<your-connection-string>
export AZURE_CONTAINER_NAME=<your-container>

# Run
python app.py --port 5000
```

### OCR Service

```bash
export OPENAI_API_KEY=<your-key>

python ocr_api.py
```

### Access the UI

Open `http://localhost:5000` in a browser to access the Materials Intelligence UI.

---

## Deployment

The application is deployed on **Azure App Service** with Gunicorn:

```bash
# startup.sh
gunicorn --bind 0.0.0.0:${PORT:-8000} --timeout 600 app:app
```

- **Timeout**: 600 seconds to accommodate long-running operations (DOCX generation, OCR processing, DINOv2 inference).
- **Workers**: Single worker (default).
- **Max upload**: 16 MB (`MAX_CONTENT_LENGTH`).

---

## Database Schema

**PostgreSQL** with the **pgvector** extension for 1024-dimensional vector similarity search.

### Core Tables

| Table | Description |
|---|---|
| `matieres` | Raw materials (reference, name, specifications) |
| `matiere_images` | Micrograph images with DINOv2 embeddings (`vector(1024)`) |
| `black_mixes` | Black Mix formulations with ADN snapshots |
| `black_mix_components` | Composition (component ref + percentage) |
| `black_mix_process_steps` | Process steps (Arbeitsfolge) |
| `black_mix_step_materials` | Step-to-material mapping |
| `nuances` | Nuance formulations with ADN snapshots |
| `nuance_images` | Nuance micrograph images with embeddings |
| `control_plans` | Control plans (sheet_data JSON: columns, measurements, statistics) |
| `fiches_adn` | ADN snapshots for raw materials |
| `cuisson_programs` | Baking programs (program ID, H₂ percentage, parameters) |

### Nested Mix Support

Components can reference raw materials **or** other Black Mixes (recursive composition):

```sql
-- XOR constraint: a component is either a matière or a sub-black-mix, never both
CHECK (
  (matiere_id IS NOT NULL AND sub_black_mix_id IS NULL)
  OR (matiere_id IS NULL AND sub_black_mix_id IS NOT NULL)
)
```

### Embedding Index

```sql
-- Cosine similarity index for fast kNN search
CREATE INDEX ON matiere_images USING ivfflat (embedding vector_cosine_ops);
```

---

## Migration Scripts

| Script | Purpose | Safety |
|---|---|---|
| `populate_database_v2.py` | Bulk import images from PPT files, compute DINOv2 embeddings, insert into DB | Checks for existing rows before insert |
| `migrate_image_paths_to_azure.py` | Convert local file paths to Azure Blob URIs | **Dry-run by default** — use `--apply` to persist changes |
| `migrate_normalize_embeddings.py` | L2-normalize all stored embedding vectors | **Idempotent** — skips already-normalized rows |
| `upload_images_to_azure.py` | Upload local images to Azure Blob Storage | Retry logic built-in, `--overwrite` flag optional |
| `upload_folders_to_azure.py` | Upload one or more local folders to Azure Blob Storage | Preserves folder structure under a target virtual folder |

### Usage

```bash
# Populate database from PPT files
python populate_database_v2.py

# Migrate paths (dry-run first, then apply)
python migrate_image_paths_to_azure.py          # dry-run
python migrate_image_paths_to_azure.py --apply   # persist

# Normalize embeddings
python migrate_normalize_embeddings.py

# Upload images to Azure
python upload_images_to_azure.py
python upload_images_to_azure.py --overwrite     # force re-upload

# Upload folders to Azure under micrograph-docs
python upload_folders_to_azure.py --target-folder micrograph-docs "Black Mix" "Data sheet" "Feuilles de contrôle" "Mix router" MSDS "Nuances concurrentes"
```

### Bulk Upload Arbitrary Folders To Azure Blob Storage

Use `upload_folders_to_azure.py` when you want to send whole local folders to Azure Blob Storage without changing the app or database.

- Required environment variables: `AZURE_CONNECTION_STRING`, `AZURE_CONTAINER_NAME`
- Optional environment variable: `AZURE_BLOB_PREFIX` (defaults to `micrograph-images`)
- Default behavior: existing blobs are skipped unless you pass `--overwrite`
- Blob layout: `{AZURE_BLOB_PREFIX}/micrograph-docs/{source-folder-name}/{relative-path-inside-source}`

Example:

```bash
python upload_folders_to_azure.py --target-folder micrograph-docs "Black Mix" "Data sheet" "Feuilles de contrôle" "Mix router" MSDS "Nuances concurrentes"
```

Useful options:

- `--overwrite` to replace blobs that already exist
- `--match text` to upload only files whose path contains the given text
- `--retries 5` to increase retry attempts on transient Azure failures

---

## Knowledge Bases

Three Markdown files serve as **domain-specific knowledge bases** referenced by the GPT Actions system prompt:

### `black_mix_kb.md` — Black Mix Workflows
- **Submit**: Extract Mischkarte (XLS) → validate all refs → POST `/black-mix/submit` with auto-ADN.
- **Update**: Full replacement via PUT `/black-mix/{id}/update`.
- **ADN Consultation**: Retrieve at three detail levels (`basic` / `combined` / `enriched`), recursively fetch component ADN.
- **DOCX Generation**: Professional reports via Groq LLM.
- **Rules**: Reference format `6600xxx` (padded), `step_materials` uses string keys (`{"1": ["ref"], "2": ["ref2"]}`), vertical Arbeitsfolge matrix read.

### `matieres_premieres_kb.md` — Raw Materials
- **OCR → Save**: PDF/XLS → extract specs → `saveOcrResult()`.
- **Specification Types**: `datasheet` (physico-chemical), `msds` (safety/GHS), `lab_control` (test parameters with min/max/target/unit).
- **ADN**: Complete material fiche with all specs and images.
- **Opportunities**: Commercial analysis for AVOCarbon application domains.

### `nuances_metalliques_kb.md` — Nuances Métalliques
- **Submit**: Mischkarte + cuisson (Wärme-Nachbehandlung) → `submitNuance()`.
- **Cuisson Programs**: Parse baking programs (e.g., `"101 25"` = program 101, 25% H₂).
- **Recursive ADN**: Walk through all component references to build complete material tree.
- **Payload Rules**: Omit `warne_nachbehandlung` and `document_revision_history` entirely if absent (never send null).

---

## Frontend UI

A single-page application served at `/` with two views:

### View 1 — Image Search
- Drag-and-drop micrograph upload zone.
- Live pipeline visualization (CLAHE → multi-scale crops → DINOv2 → L2-norm → pgvector).
- Result cards with similarity score bars.
- Inline ADN display and DOCX download.

### View 2 — Compare ADN
- Parallel upload zones (Image A / VS / Image B).
- Independent search for each image.
- Cross-embedding cosine similarity computation.
- Groq-powered differential analysis displayed in tabs: **Differences** | **ADN-A** | **ADN-B**.

**Technology**: Vanilla JavaScript (zero dependencies), CSS custom properties design system, Jinja2 template.

**Fonts**: Syne (display), DM Sans (body), DM Mono (monospace).

---

## Key Design Patterns

| Pattern | Details |
|---|---|
| **Multi-Scale Embeddings** | 4 crop views (100%, 92%, 80%, 66%) averaged and L2-normalized for robust similarity |
| **Reference Padding** | All numeric references → `6600{padded}` (e.g., `35` → `6600035`) |
| **Vertical Matrix Read** | `step_materials` extracted cell-by-cell from Arbeitsfolge grid, never inferred |
| **ADN Single-Line Storage** | ADN snapshot stored as single TEXT row (overwrite model, not versioned) |
| **Optional Field Omission** | Never send `null` for optional strings; omit the field entirely if absent |
| **Component Type Resolution** | References validated against three types: matière, black_mix, nuance |
| **Idempotent Migrations** | All scripts check existence, support dry-run, skip already-processed rows |
| **Groq Key Rotation** | Multiple API keys with round-robin rotation for rate limit resilience |
| **Temp File Cleanup** | Background thread removes files older than 30 minutes every 5 minutes |

---

## Technical Specifications

| Spec | Value |
|---|---|
| Embedding Model | DINOv2-large (CLS token) |
| Embedding Dimension | 1024 |
| Vector Normalization | L2 (cosine similarity) |
| LLM (ADN/DOCX) | Groq Llama 3.3 70B |
| LLM (Analysis) | OpenAI GPT-4o |
| Max Upload Size | 16 MB |
| Gunicorn Timeout | 600 seconds |
| Image Formats | PNG, JPG, JPEG, WEBP |
| Supported Documents | PDF, XLS, XLSX, PPT, PPTX |


