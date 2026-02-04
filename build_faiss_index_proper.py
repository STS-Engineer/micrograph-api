"""
Build FAISS index from pre-computed embeddings.

Usage:
    python build_faiss_index_proper.py --embeddings_dir ./faiss_v3 --model dinov2
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
import faiss


def build_faiss_index(embeddings_path: str, output_dir: str, model_name: str):
    """Build and save FAISS index from embeddings."""
    
    print(f"📖 Loading embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"   Shape: {embeddings.shape}")
    
    # Normalize embeddings for cosine similarity
    print(f"🔄 Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index
    print(f"🏗️  Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"✅ Index built with {index.ntotal} vectors")
    
    # Save index
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    index_file = output_path / f"faiss_index_{model_name}.index"
    faiss.write_index(index, str(index_file))
    print(f"💾 FAISS index saved to: {index_file}")
    
    # Create search config
    metadata_file = output_path / f"metadata_{model_name}.json"
    embeddings_file = output_path / f"embeddings_{model_name}.npy"
    
    config = {
        "model": model_name,
        "index_file": f"faiss_index_{model_name}.index",
        "metadata_file": f"metadata_{model_name}.json",
        "embeddings_file": f"embeddings_{model_name}.npy",
        "index_type": "IndexFlatIP",
        "dimension": int(dimension),
        "num_vectors": int(index.ntotal)
    }
    
    config_file = output_path / f"search_config_{model_name}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"📄 Search config saved to: {config_file}")
    
    print(f"\n🎉 Done! Use this config file for searching:")
    print(f"   {config_file}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--embeddings_dir", type=str, required=True,
                       help="Directory containing embeddings and metadata")
    parser.add_argument("--model", type=str, default="dinov2",
                       choices=["dinov2", "clip"],
                       help="Model name (used for file naming)")
    
    args = parser.parse_args()
    
    embeddings_path = Path(args.embeddings_dir) / f"embeddings_{args.model}.npy"
    
    if not embeddings_path.exists():
        print(f"❌ Embeddings file not found: {embeddings_path}")
        return
    
    build_faiss_index(
        embeddings_path=str(embeddings_path),
        output_dir=args.embeddings_dir,
        model_name=args.model
    )


if __name__ == "__main__":
    main()
