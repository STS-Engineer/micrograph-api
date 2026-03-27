from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv


load_dotenv()

DB_DSN = os.getenv(
    "DB_DSN",
    "postgresql://administrationSTS:St%24%400987@avo-adb-002.postgres.database.azure.com:5432/Micrographie_IA",
).strip()
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "").strip()
AZURE_BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "micrograph-images").strip("/ ")

TABLES = [
    ("public.matiere_images", "id", "image_path"),
    ("public.nuance_images", "id", "image_path"),
]


def build_blob_storage_path(source_path: str) -> Optional[str]:
    normalized = (source_path or "").replace("\\", "/").strip()
    if not normalized:
        return None
    if normalized.startswith("azure-blob://"):
        return normalized

    parts = Path(normalized).parts
    if not parts:
        return None

    if "output_v3" in parts:
        filename = Path(normalized).name
        blob_name = f"{AZURE_BLOB_PREFIX}/output_v3/{filename}" if AZURE_BLOB_PREFIX else f"output_v3/{filename}"
    elif "output_v4" in parts:
        filename = Path(normalized).name
        blob_name = f"{AZURE_BLOB_PREFIX}/output_v4/{filename}" if AZURE_BLOB_PREFIX else f"output_v4/{filename}"
    elif "embeddings_v7" in parts:
        filename = Path(normalized).name
        blob_name = f"{AZURE_BLOB_PREFIX}/embeddings_v7/{filename}" if AZURE_BLOB_PREFIX else f"embeddings_v7/{filename}"
    else:
        filename = Path(normalized).name
        blob_name = f"{AZURE_BLOB_PREFIX}/{filename}" if AZURE_BLOB_PREFIX else filename

    return f"azure-blob://{AZURE_CONTAINER_NAME}/{blob_name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate local image_path values to Azure blob storage paths.")
    parser.add_argument("--apply", action="store_true", help="Persist the updates. Without this flag, runs as dry-run.")
    args = parser.parse_args()

    if not AZURE_CONTAINER_NAME:
        raise RuntimeError("Missing required environment variable: AZURE_CONTAINER_NAME")

    conn = psycopg2.connect(DB_DSN)
    try:
        total_candidates = 0
        total_updates = 0
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for table_name, id_column, path_column in TABLES:
                cur.execute(f"SELECT {id_column}, {path_column} FROM {table_name} ORDER BY {id_column} ASC")
                rows = cur.fetchall()
                table_candidates = 0
                table_updates = 0

                for row in rows:
                    row_id = row[id_column]
                    current_path = row[path_column]
                    target_path = build_blob_storage_path(current_path)
                    if not target_path or target_path == current_path:
                        continue

                    table_candidates += 1
                    total_candidates += 1
                    print(f"[{table_name}] id={row_id}")
                    print(f"  old: {current_path}")
                    print(f"  new: {target_path}")

                    if args.apply:
                        cur.execute(
                            f"UPDATE {table_name} SET {path_column} = %s WHERE {id_column} = %s",
                            (target_path, row_id),
                        )
                        table_updates += 1
                        total_updates += 1

                print(f"{table_name}: candidates={table_candidates}, updated={table_updates if args.apply else 0}")

        if args.apply:
            conn.commit()
            print(f"Committed updates. total_candidates={total_candidates}, total_updated={total_updates}")
        else:
            conn.rollback()
            print(f"Dry run complete. total_candidates={total_candidates}, total_updated=0")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
