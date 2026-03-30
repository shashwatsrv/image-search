from PIL import Image
import os

def load_image(image_path:str)-> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image=Image.open(image_path).convert("RGB")
    return image

def get_image_paths(folder_path: str) -> list:
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = []

    for root, _, files in os.walk(folder_path):
        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_exts:
                image_paths.append(os.path.join(root, filename))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in folder: {folder_path}")

    return sorted(image_paths)