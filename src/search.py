import faiss
import numpy as np
import pandas as pd

from models.embedder import embed_image, embed_text
from src.index import load_index
from PIL import Image


class ImageSearchEngine:
    def __init__(self, index_type: str = "hnsw"):
        self.index_type = index_type

        # load once (IMPORTANT: no per-query loading)
        self.index = load_index(f"data/index_{index_type}.faiss")
        self.metadata = pd.read_csv(f"data/metadata_{index_type}.csv")
        self.embeddings = np.load(f"data/embeddings_{index_type}.npy")

        # ensure numpy float32 (FAISS compatibility)
        self.embeddings = self.embeddings.astype("float32")

        print(f"[INFO] Search engine loaded: {self.index.ntotal} vectors")

    def _embed_query(self, query):
        if isinstance(query, str):
            # text OR file path logic (your embedder handles both)
            return embed_text(query)

        if isinstance(query, Image.Image):
            return embed_image(query)

        raise ValueError("Query must be a string or PIL Image")

    def search(self, query, k: int = 5):
        # ---- embed query ----
        vector = self._embed_query(query).astype("float32").reshape(1, -1)


        # HNSW tuning (if used)
        if isinstance(self.index, faiss.IndexHNSWFlat):
            self.index.hnsw.efSearch = 256

        # retrieve more candidates for reranking
        top_k = max(50, k * 10)
        scores, indices = self.index.search(vector, top_k)

        indices = indices[0]

        # filter invalid indices (safety)
        valid_mask = indices != -1
        indices = indices[valid_mask]

        if len(indices) == 0:
            return []

        # ---- rerank using original embeddings ----
        candidates = self.embeddings[indices]  # (N, D)
        rerank_scores = candidates @ vector.T   # cosine similarity (already normalized)

        rerank_scores = rerank_scores.squeeze()
        order = np.argsort(rerank_scores)[::-1]

        # ---- build results ----
        results = []
        for rank in order[:k]:
            idx = indices[rank]

            if idx >= len(self.metadata):
                continue

            row = self.metadata.iloc[idx]

            results.append({
                "score": float(rerank_scores[rank]),
                "image_path": row.get("image_path", ""),
                "class_name": row.get("class_name", "")
            })

        return results