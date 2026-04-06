from __future__ import annotations

import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.core.exceptions import ServiceResponseError
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
AZURE_CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING", "").strip()
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME", "").strip()
AZURE_BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "micrograph-images").strip("/ ")
DEFAULT_TARGET_FOLDER = "micrograph-docs"


def require_env(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def resolve_source_dir(source_dir: str) -> Path:
    candidate = Path(source_dir).expanduser()
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate.resolve()


def build_blob_name(source_dir: Path, file_path: Path, target_folder: str) -> str:
    relative_name = file_path.relative_to(source_dir).as_posix()
    source_folder_name = source_dir.name
    if AZURE_BLOB_PREFIX:
        return f"{AZURE_BLOB_PREFIX}/{target_folder}/{source_folder_name}/{relative_name}"
    return f"{target_folder}/{source_folder_name}/{relative_name}"


def iter_source_files(source_dir: Path, match_filter: str):
    for file_path in source_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if match_filter and match_filter not in str(file_path).casefold():
            continue
        yield file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload one or more local folders to Azure Blob Storage while preserving their structure."
    )
    parser.add_argument(
        "source_dirs",
        nargs="+",
        help="One or more local folders to upload.",
    )
    parser.add_argument(
        "--target-folder",
        default=DEFAULT_TARGET_FOLDER,
        help="Virtual folder inside the Azure blob prefix.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite blobs that already exist in Azure.",
    )
    parser.add_argument(
        "--match",
        default="",
        help="Only process files whose full path contains this text (case-insensitive).",
    )
    parser.add_argument(
        "--connection-timeout",
        type=int,
        default=180,
        help="Client-side socket write/connect timeout in seconds for each Azure request.",
    )
    parser.add_argument(
        "--read-timeout",
        type=int,
        default=180,
        help="Client-side socket read timeout in seconds for each Azure request.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of attempts for each blob upload on transient network errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    require_env("AZURE_CONNECTION_STRING", AZURE_CONNECTION_STRING)
    require_env("AZURE_CONTAINER_NAME", AZURE_CONTAINER_NAME)

    target_folder = args.target_folder.strip("/ ") or DEFAULT_TARGET_FOLDER
    match_filter = args.match.casefold().strip()
    source_dirs = [resolve_source_dir(source_dir) for source_dir in args.source_dirs]

    for source_dir in source_dirs:
        if not source_dir.exists():
            raise RuntimeError(f"Source directory does not exist: {source_dir}")
        if not source_dir.is_dir():
            raise RuntimeError(f"Source path is not a directory: {source_dir}")

    blob_service = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    uploaded = 0
    overwritten = 0
    skipped_existing = 0
    failed = 0
    scanned = 0

    for source_dir in source_dirs:
        print(f"Scanning: {source_dir}")
        for file_path in iter_source_files(source_dir, match_filter):
            scanned += 1
            blob_name = build_blob_name(source_dir, file_path, target_folder)
            blob_client = container_client.get_blob_client(blob_name)
            blob_exists = blob_client.exists()

            if blob_exists and not args.overwrite:
                skipped_existing += 1
                continue

            content_type, _ = mimetypes.guess_type(file_path.name)
            print(f"Uploading: {file_path} -> {blob_name}")

            upload_succeeded = False
            for attempt in range(1, max(args.retries, 1) + 1):
                try:
                    with open(file_path, "rb") as stream:
                        blob_client.upload_blob(
                            stream,
                            overwrite=args.overwrite,
                            content_type=content_type or "application/octet-stream",
                            connection_timeout=args.connection_timeout,
                            read_timeout=args.read_timeout,
                        )
                    upload_succeeded = True
                    break
                except ServiceResponseError as exc:
                    print(
                        f"Attempt {attempt}/{max(args.retries, 1)} failed for {file_path.name}: {exc}"
                    )
                    if attempt < max(args.retries, 1):
                        time.sleep(min(2 ** (attempt - 1), 10))
                    else:
                        failed += 1
                except Exception as exc:
                    print(f"Failed: {file_path} -> {blob_name} ({type(exc).__name__}: {exc})")
                    failed += 1
                    break

            if not upload_succeeded:
                continue

            if blob_exists:
                overwritten += 1
                print(f"Overwritten: {file_path} -> {blob_name}")
            else:
                uploaded += 1
                print(f"Uploaded: {file_path} -> {blob_name}")

    print(
        "Done. "
        f"Scanned={scanned}, Uploaded={uploaded}, Overwritten={overwritten}, "
        f"SkippedExisting={skipped_existing}, Failed={failed}, "
        f"Container={AZURE_CONTAINER_NAME}, TargetFolder={target_folder}"
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()