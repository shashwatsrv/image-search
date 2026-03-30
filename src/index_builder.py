import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from models.embedder import embed_image
from src.utils import load_image, get_image_paths
from src.index import build_index, save_index


DATA_DIR = "data/intel"           # change if needed
INDEX_PATH = "data/index.faiss"
METADATA_PATH = "data/metadata.csv"


def build_pipeline(data_dir: str = DATA_DIR):
    print("[INFO] Scanning images...")
    image_paths = get_image_paths(data_dir)

    print(f"[INFO] Found {len(image_paths)} images")

    embeddings = []
    valid_paths = []

    print("[INFO] Generating embeddings...")

    for path in tqdm(image_paths):
        try:
            image = load_image(path)
            vec = embed_image(image)

            embeddings.append(vec)
            valid_paths.append(path)

        except Exception as e:
            print(f"[WARNING] Skipping {path}: {e}")

    embeddings = np.vstack(embeddings).astype("float32")

    print("[INFO] Building FAISS index...")
    index = build_index(embeddings)

    print("[INFO] Saving index...")
    save_index(index, INDEX_PATH)

    print("[INFO] Creating metadata...")

    metadata = pd.DataFrame({
        "image_path": valid_paths,
        "class_name": [os.path.basename(os.path.dirname(p)) for p in valid_paths]
    })

    metadata.to_csv(METADATA_PATH, index=False)

    print(f"[INFO] Metadata saved to {METADATA_PATH}")
    print("[INFO] Pipeline complete ✅")


if __name__ == "__main__":
    build_pipeline()