import faiss
import numpy as np
import os


def build_index(embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D (N, D)")

    embeddings = embeddings.astype("float32")


    d = embeddings.shape[1]

    # create index
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)

    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32)
        index.hnsw.efConstruction = 200

    else:
        raise ValueError("Unknown index type")

    # add ONCE
    index.add(embeddings)

    print(f"[INFO] Indexed {index.ntotal} vectors with dimension {d}")

    return index


def save_index(index: faiss.Index, path: str = "data/index.faiss"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    print(f"[INFO] Index saved at {path}")


def load_index(path: str = "data/index.faiss") -> faiss.Index:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Index file not found at {path}")

    index = faiss.read_index(path)
    print(f"[INFO] Loaded index with {index.ntotal} vectors")

    return index