import faiss
import numpy as np
import pandas as pd
from models.embedder import embed_image, embed_text
from src.index import load_index
from PIL import Image

def search(query:Image.Image | str,k:int=5):
    index=load_index()
    metadata=pd.read_csv("data/metadata.csv")

    if isinstance(query,str):
        vector=embed_text(query)
    else:
        vector=embed_image(query)

    vector = vector.reshape(1, -1).astype("float32")
    distances,indices=index.search(vector,k)

    results=[]
    for idx,score in zip(indices[0],distances[0]):
        row=metadata.iloc[idx]
        results.append({
    "index": int(idx),
    "score": float(score),
    "class_name": row.get("class_name", ""),
    "image_path": row.get("image_path", "")
    })
    return results