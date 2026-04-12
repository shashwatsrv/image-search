import torch
import os
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from src.utils import load_image

MODEL_NAME = "openai/clip-vit-base-patch16"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / (vector.norm(dim=-1, keepdim=True) + 1e-8)

def embed_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    inputs = processor(images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vector = model.get_image_features(**inputs)

    vector = normalize(vector)

    return vector.squeeze().cpu().numpy().astype("float32")


def embed_text(text: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vector = model.get_text_features(**inputs)

    vector = normalize(vector)

    return vector.squeeze().cpu().numpy().astype("float32")

def embed_images_batch(images) -> np.ndarray:
    images = [img.convert("RGB") for img in images]

    inputs = processor(images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        vectors = model.get_image_features(**inputs)

    vectors = normalize(vectors)

    
def embed_query(query):
    if isinstance(query, str):
        if os.path.exists(query):
            img=load_image(query)
            return embed_image(img)
        else:
            return embed_text(query)
    elif isinstance(query, Image.Image):
        return embed_image(query)
    else:
        raise ValueError("Query must be a string or a PIL Image.")