"""
Compute DINOv2 embeddings for all reference micrographs.

Usage:
    python compute_embeddings.py --metadata_path ./references/metadata.json --output_dir ./embeddings --model dinov2
Optional:
    python compute_embeddings.py --metadata_path ./references/metadata.json --output_dir ./embeddings --images_root ./references/images
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

import torch
from PIL import Image
from tqdm import tqdm


def resolve_image_path(image_path_str: str, metadata_path: str, images_root: Optional[str] = None) -> Optional[str]:
    """
    Resolve image path robustly:
    Priority:
      1) absolute path (if exists)
      2) relative to metadata directory
      3) images_root + basename
      4) images_root + relative path (if metadata stores something like references_v4\\images\\...)
      5) as-is relative to current working directory
    Returns:
      resolved absolute path as str if found, else None
    """
    if not image_path_str:
        return None

    # Normalize separators
    image_path_str = image_path_str.replace("/", os.sep).replace("\\", os.sep)
    p = Path(image_path_str)

    # 1) absolute and exists
    if p.is_absolute() and p.exists():
        return str(p.resolve())

    meta_dir = Path(metadata_path).resolve().parent

    # 2) relative to metadata dir
    cand = (meta_dir / p).resolve()
    if cand.exists():
        return str(cand)

    # 3) images_root + basename
    if images_root:
        root = Path(images_root).resolve()
        cand2 = (root / p.name).resolve()
        if cand2.exists():
            return str(cand2)

        # 4) images_root + full relative path
        cand3 = (root / p).resolve()
        if cand3.exists():
            return str(cand3)

    # 5) current working directory fallback
    cand4 = (Path.cwd() / p).resolve()
    if cand4.exists():
        return str(cand4)

    return None


class EmbeddingComputer:
    """Compute image embeddings using DINOv2 or CLIP."""

    def __init__(self, model_name: str = "dinov2", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔧 Loading {model_name} model on {self.device}...")

        if model_name == "dinov2":
            self._load_dinov2()
        elif model_name == "clip":
            self._load_clip()
        else:
            raise ValueError(f"Unknown model: {model_name}")

        print("✅ Model loaded successfully")

    def _load_dinov2(self):
        from transformers import AutoModel, AutoImageProcessor

        model_id = "facebook/dinov2-large"
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoImageProcessor.from_pretrained(model_id)

        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = 1024  # dinov2-large

    def _load_clip(self):
        from transformers import CLIPModel, CLIPProcessor

        model_id = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = 768  # clip vit-l/14

    def compute_embedding(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert("RGB")

            if self.model_name == "dinov2":
                return self._compute_dinov2_embedding(image)
            elif self.model_name == "clip":
                return self._compute_clip_embedding(image)

        except Exception as e:
            print(f"⚠️  Error computing embedding for {image_path}: {e}")
            return None

    def _compute_dinov2_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()  # CLS token
            embedding = embedding.cpu().numpy()

        return embedding

    def _compute_clip_embedding(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            embedding = image_features.squeeze().cpu().numpy()

        return embedding

    def compute_batch(self, metadata_list: List[Dict], metadata_path: str, images_root: Optional[str] = None) -> tuple:
        """
        Returns:
            (embeddings_array, valid_metadata)
        """
        embeddings: List[np.ndarray] = []
        valid_metadata: List[Dict] = []

        print(f"\n🔄 Computing embeddings for {len(metadata_list)} images...")

        for meta in tqdm(metadata_list):
            raw_path = meta.get("image_path", "")
            resolved = resolve_image_path(raw_path, metadata_path=metadata_path, images_root=images_root)

            if not resolved or not os.path.exists(resolved):
                print(f"⚠️  Image not found: {raw_path}")
                continue

            embedding = self.compute_embedding(resolved)
            if embedding is None:
                continue

            # store resolved absolute path to avoid future mismatch
            meta = dict(meta)  # copy to avoid mutating original unexpectedly
            meta["image_path"] = resolved

            embeddings.append(embedding)
            valid_metadata.append(meta)

        if len(embeddings) == 0:
            embeddings_array = np.zeros((0,), dtype=np.float32)
        else:
            embeddings_array = np.array(embeddings, dtype=np.float32)

        print(f"✅ Computed {len(embeddings_array)} embeddings")
        print(f"   Shape: {embeddings_array.shape}")

        return embeddings_array, valid_metadata


def save_embeddings(embeddings: np.ndarray, metadata: List[Dict], output_dir: str, model_name: str):
    """Save embeddings and metadata safely."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # If no embeddings, stop with clear message
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise RuntimeError(
            "❌ 0 embeddings calculés.\n"
            "➡️ Vérifie que metadata.json pointe vers le bon dossier d’images.\n"
            "➡️ Astuce: passe --images_root ./references_v3/images (ou references_v4/images) pour forcer le bon dossier."
        )

    embeddings_file = output_path / f"embeddings_{model_name}.npy"
    np.save(embeddings_file, embeddings)
    print(f"💾 Embeddings saved to: {embeddings_file}")

    # Add embedding index to metadata
    for idx, meta in enumerate(metadata):
        meta["embedding_index"] = idx
        meta["model"] = model_name

    metadata_file = output_path / f"metadata_{model_name}.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"📄 Metadata saved to: {metadata_file}")

    # Statistics
    norms = np.linalg.norm(embeddings, axis=1)
    stats = {
        "model": model_name,
        "num_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std()),
    }

    stats_file = output_path / f"stats_{model_name}.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"📊 Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings for micrographs")
    parser.add_argument("--metadata_path", type=str, required=True, help="Path to metadata.json from extraction step")
    parser.add_argument("--output_dir", type=str, default="./embeddings", help="Output directory for embeddings")
    parser.add_argument("--model", type=str, default="dinov2", choices=["dinov2", "clip"], help="Embedding model to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu, auto-detected if not specified)")
    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Optional: Force images root directory (e.g., ./references_v3/images). Useful if metadata paths are inconsistent.",
    )

    args = parser.parse_args()

    print(f"📖 Loading metadata from: {args.metadata_path}")
    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"   Found {len(metadata)} entries")

    computer = EmbeddingComputer(model_name=args.model, device=args.device)
    embeddings, valid_metadata = computer.compute_batch(metadata, metadata_path=args.metadata_path, images_root=args.images_root)

    save_embeddings(embeddings, valid_metadata, args.output_dir, args.model)
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
