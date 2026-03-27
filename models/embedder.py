import torch
import numpy as np

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch16"

device="cuda" if torch.cuda.is_available() else "cpu"

model=CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor=CLIPProcessor.from_pretrained(MODEL_NAME)

def embed_image(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        vector=model.get_image_features(**inputs)
    vector=vector/vector.norm(dim=1,keepdim=True)
    return vector.squeeze().cpu().numpy()

def embed_text(text:str) -> np.ndarray:
    inputs = processor(text=text, return_tensors="pt",padding=True)
    inputs={k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        vector=model.get_text_features(**inputs)
    vector=vector/vector.norm(dim=1,keepdim=True)
    return vector.squeeze().cpu().numpy()