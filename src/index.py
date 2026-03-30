import faiss
import numpy as np
import os


def build_index(embeddings: np.ndarray) -> faiss.Index:

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array of shape (N, D)")

    #ensure correct dtype
    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"[INFO] Indexed {index.ntotal} vectors with dimension {dimension}")

    return index


def save_index(index: faiss.Index, path: str = "data/index.faiss"):
#save faiss index to disk
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"[INFO] Index saved at {path}")


def load_index(path: str = "data/index.faiss") -> faiss.Index:
#load faiss index from disk
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found at {path}")

    index = faiss.read_index(path)
    print(f"[INFO] Loaded index with {index.ntotal} vectors")

    return index