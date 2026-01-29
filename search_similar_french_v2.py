#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import torch
from PIL import Image
import faiss
import re


class FrenchMicrographSearchEngine:
    def __init__(self, config_path: str):
        print("🔧 Initializing search engine...")

        self.config_path = str(Path(config_path).resolve())
        self.config_dir = str(Path(self.config_path).parent.resolve())

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # ================================
        # ✅ FIX: robust path resolution
        # ================================
        def _resolve_path(p: str) -> str:
            p = str(p).strip().strip('"').strip("'")

            if os.path.isabs(p):
                return p

            # 1) relative to config directory
            cand1 = Path(self.config_dir) / p
            if cand1.exists():
                return str(cand1.resolve())

            # 2) relative to current working directory
            cand2 = Path(os.getcwd()) / p
            if cand2.exists():
                return str(cand2.resolve())

            # 3) fallback (clear error path)
            return str(cand1.resolve())

        self.config["index_file"] = _resolve_path(self.config["index_file"])
        self.config["metadata_file"] = _resolve_path(self.config["metadata_file"])
        # ================================

        print("📖 Loading FAISS index...")
        self.index = faiss.read_index(self.config["index_file"])
        print(f"   Loaded {self.index.ntotal} vectors")

        print("📖 Loading metadata...")
        with open(self.config["metadata_file"], "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"   Loaded {len(self.metadata)} entries")

        print(f"🔧 Loading model: {self.config['model']}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

        print("✅ Search engine ready!")

    def _load_model(self):
        if self.config["model"] == "dinov2":
            from transformers import AutoModel, AutoImageProcessor
            model_id = "facebook/dinov2-large"
            self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
            self.processor = AutoImageProcessor.from_pretrained(model_id)

        elif self.config["model"] == "clip":
            from transformers import CLIPModel, CLIPProcessor
            model_id = "openai/clip-vit-large-patch14"
            self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
            self.processor = CLIPProcessor.from_pretrained(model_id)

        else:
            raise ValueError(f"Unknown model: {self.config['model']}")

    def compute_query_embedding(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        return self.compute_query_embedding_from_pil(image)

    def compute_query_embedding_from_pil(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")

        if self.config["model"] == "dinov2":
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

        else:  # CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model.get_image_features(**inputs).squeeze().cpu().numpy()

        return emb.astype("float32")

    def search(self, query_image_path: str, top_k: int = 1) -> List[Dict]:
        query_embedding = self.compute_query_embedding(query_image_path)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta["similarity_score"] = float(dist)
            results.append(meta)

        return results

    def search_from_pil(self, image: Image.Image, top_k: int = 1) -> List[Dict]:
        query_embedding = self.compute_query_embedding_from_pil(image)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx].copy()
            meta["similarity_score"] = float(dist)
            results.append(meta)

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()

    engine = FrenchMicrographSearchEngine(args.config)
    results = engine.search(args.query, args.top_k)

    for r in results:
        print(r["similarity_score"], r.get("reference", ""))


if __name__ == "__main__":
    main()
