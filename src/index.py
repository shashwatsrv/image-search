import faiss
import numpy as np  

def build_index(embeddings:np.ndarray) -> faiss.Index:
    dimension=embeddings.shape[1] #512
    index=faiss.IndexFlatIP(dimension)  #Inner Product for cosine similarity
    index.add(embeddings)  #Add all embeddings to the index
    return index

def save_index(index:faiss.Index,path:str="data/index.faiss"):
    faiss.write_index(index,path)

def load_index(path:str="data/index.faiss")->faiss.Index:
    return faiss.read_index(path)