from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING", "").strip()
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "").strip()
AZURE_BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "micrograph-images").strip("/ ")

IMAGE_DIRS = [
    BASE_DIR / "embeddings_v7" / "images",
    BASE_DIR / "output_v3" / "images",
    BASE_DIR / "output_v4" / "images",
]


def require_env(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_blob_name(source_dir: Path, file_path: Path) -> str:
    relative_name = file_path.relative_to(source_dir).as_posix()
    folder_name = source_dir.parent.name
    if AZURE_BLOB_PREFIX:
        return f"{AZURE_BLOB_PREFIX}/{folder_name}/{relative_name}"
    return f"{folder_name}/{relative_name}"


def main() -> None:
    require_env("AZURE_CONNECTION_STRING", AZURE_CONNECTION_STRING)
    require_env("AZURE_CONTAINER_NAME", AZURE_CONTAINER_NAME)

    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    uploaded = 0
    skipped = 0

    for image_dir in IMAGE_DIRS:
        if not image_dir.exists():
            print(f"Skipping missing directory: {image_dir}")
            continue

        for file_path in image_dir.rglob("*"):
            if not file_path.is_file():
                continue

            blob_name = build_blob_name(image_dir, file_path)
            blob_client = container_client.get_blob_client(blob_name)

            if blob_client.exists():
                skipped += 1
                continue

            content_type, _ = mimetypes.guess_type(file_path.name)
            with open(file_path, "rb") as f:
                blob_client.upload_blob(
                    f,
                    overwrite=False,
                    content_type=content_type or "application/octet-stream",
                )
            uploaded += 1
            print(f"Uploaded: {file_path} -> {blob_name}")

    print(f"Done. Uploaded={uploaded}, Skipped={skipped}, Container={AZURE_CONTAINER_NAME}")


if __name__ == "__main__":
    main()
