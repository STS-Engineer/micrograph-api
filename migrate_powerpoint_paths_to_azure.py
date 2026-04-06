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
AZURE_PPT_TARGET_FOLDER = os.getenv("AZURE_PPT_TARGET_FOLDER", "micrographie-ppts-inputs").strip("/ ")

TABLE_NAME = "public.powerpoint_files"
ID_COLUMN = "id"
PATH_COLUMN = "file_path"


def build_blob_storage_path(source_path: str) -> Optional[str]:
    normalized = (source_path or "").replace("\\", "/").strip()
    if not normalized:
        return None
    if normalized.startswith("azure-blob://"):
        return normalized

    filename = Path(normalized).name
    if not filename:
        return None

    blob_name = f"{AZURE_BLOB_PREFIX}/{AZURE_PPT_TARGET_FOLDER}/{filename}" if AZURE_BLOB_PREFIX else f"{AZURE_PPT_TARGET_FOLDER}/{filename}"
    return f"azure-blob://{AZURE_CONTAINER_NAME}/{blob_name}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate powerpoint_files.file_path values to Azure blob storage paths.")
    parser.add_argument("--apply", action="store_true", help="Persist the updates. Without this flag, runs as dry-run.")
    args = parser.parse_args()

    if not AZURE_CONTAINER_NAME:
        raise RuntimeError("Missing required environment variable: AZURE_CONTAINER_NAME")

    conn = psycopg2.connect(DB_DSN)
    try:
        total_candidates = 0
        total_updates = 0
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT {ID_COLUMN}, {PATH_COLUMN} FROM {TABLE_NAME} ORDER BY {ID_COLUMN} ASC")
            rows = cur.fetchall()

            for row in rows:
                row_id = row[ID_COLUMN]
                current_path = row[PATH_COLUMN]
                target_path = build_blob_storage_path(current_path)
                if not target_path or target_path == current_path:
                    continue

                total_candidates += 1
                print(f"[{TABLE_NAME}] id={row_id}")
                print(f"  old: {current_path}")
                print(f"  new: {target_path}")

                if args.apply:
                    cur.execute(
                        f"UPDATE {TABLE_NAME} SET {PATH_COLUMN} = %s WHERE {ID_COLUMN} = %s",
                        (target_path, row_id),
                    )
                    total_updates += 1

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