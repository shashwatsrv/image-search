import faiss
import numpy as np
import pandas as pd
from models.embedder import embed_image, embed_text
from src.index import load_index
from PIL import Image


def search(query: Image.Image | str, k: int = 5, index_type: str = "hnsw"):

    index = load_index(f"data/index_{index_type}.faiss")
    metadata = pd.read_csv(f"data/metadata_{index_type}.csv")
    embeddings = np.load(f"data/embeddings_{index_type}.npy")

    # embed query
    if isinstance(query, str):
        vector = embed_text(query)
    else:
        vector = embed_image(query)

    vector = vector.reshape(1, -1).astype("float32")
    faiss.normalize_L2(vector)

    # HNSW tuning
    if isinstance(index, faiss.IndexHNSWFlat):
        index.hnsw.efSearch = 256

    # retrieve more candidates
    top_k = max(50, k * 10)
    scores, indices = index.search(vector, top_k)

    # rerank
    candidates = embeddings[indices[0]]
    rerank_scores = (candidates @ vector.T).squeeze()

    order = np.argsort(rerank_scores)[::-1]

    indices = indices[0][order]
    rerank_scores = rerank_scores[order]

    # results
    #THRESHOLD = 0.3 

    results = []
    for i, idx in enumerate(indices[:k]):

        # if rerank_scores[i] < THRESHOLD:
        #     continue

        row = metadata.iloc[idx]

        results.append({
            "index": int(idx),
            "score": float(rerank_scores[i]),
            "class_name": row.get("class_name", ""),
            "image_path": row.get("image_path", "")
        })
    return results