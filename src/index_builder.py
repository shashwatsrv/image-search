import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse #cli driven index building

from models.embedder import embed_images_batch
from src.utils import load_image, get_image_paths
from src.index import build_index, save_index



DATA_DIR = "data/intel/seg_test"           #change if needed
BATCH_SIZE = 32 

def build_pipeline(data_dir: str = DATA_DIR, index_type: str = "flat"):

    INDEX_PATH = f"data/index_{index_type}.faiss"
    METADATA_PATH = f"data/metadata_{index_type}.csv"


    print("[INFO] Scanning images...")
    image_paths = get_image_paths(data_dir)

    print(f"[INFO] Found {len(image_paths)} images")

    embeddings = []
    valid_paths = []

    print("[INFO] Generating embeddings...")

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        images = []
        batch_valid_paths = [] 
        for p in batch_paths:
            try:
                img=load_image(p)
                images.append(img)
                batch_valid_paths.append(p)
            except Exception as e:
                print(f"[WARNING] Failed to load {p}: {e}")
        if len(images) == 0:
            continue
        vecs = embed_images_batch(images)
        embeddings.append(vecs)
        valid_paths.extend(batch_valid_paths)
                
    embeddings = np.vstack(embeddings).astype("float32")
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    np.save("data/embeddings.npy", embeddings) #save embeddings for later use (hnsw)

    print("[INFO] Building FAISS index...")
    index = build_index(embeddings, index_type=index_type)

    print("[INFO] Saving index...")
    save_index(index, INDEX_PATH)

    print("[INFO] Creating metadata...")

    metadata = pd.DataFrame({
        "image_path": valid_paths,
        "class_name": [os.path.normpath(p).split(os.sep)[-2] for p in valid_paths]
    })

    metadata.to_csv(METADATA_PATH, index=False)

    print(f"[INFO] Metadata saved to {METADATA_PATH}")
    print("[INFO] Pipeline complete ✅")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="flat", choices=["flat", "hnsw"])
    args = parser.parse_args()

    build_pipeline(index_type=args.index)