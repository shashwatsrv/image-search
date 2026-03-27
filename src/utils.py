from PIL import image
import os

def load_image(image_path:str)-> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFound(f"Image not found: {image_path}")
    
    image=Image.open(image_path).convert("RGB")
    return image

def get_image_paths(folder_path:str)-> list:
    #supported img formats
    valid_exts={".jpg",".jpeg",".png",".webp",".bmp"}
    image_paths=[]

    for fname in os.listdir(folder_path):
        ext=os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            full_path=os.path.join(folder_path,fname)
            image_paths.append(full_path)

        if len(image_paths)==0:
            raise ValueError(f"No image found in folder: {folder_path}")

    return sorted(image_path)

