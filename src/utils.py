from PIL import Image, ImageFile
import os
import aiohttp
import io
import asyncio

# handle partially downloaded images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_image(image_path: str) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = Image.open(image_path).convert("RGB")

    # resize for consistency
    img.thumbnail((512, 512))

    return img


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


async def load_image_from_url(session, url, retries=3):
    for _ in range(retries):
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    continue

                data = await response.read()

                img = Image.open(io.BytesIO(data)).convert("RGB")

                # ---- critical fixes ----

                # skip extremely large images
                if img.width * img.height > 50_000_000:
                    return None

                # resize to consistent size
                img.thumbnail((512, 512))

                return img

        except Exception:
            await asyncio.sleep(0.3)

    return None