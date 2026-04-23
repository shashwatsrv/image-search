from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.search import ImageSearchEngine


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"

app = FastAPI(title="Image Search Web App")
search_engine = ImageSearchEngine(index_type="hnsw")


def normalize_image_path(path_str: str) -> Path:
    cleaned = path_str.replace("\\", "/")
    path = Path(cleaned)
    return (ROOT_DIR / path).resolve()


@app.get("/")
def serve_frontend():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_file)


@app.get("/api/search")
def search_images(
    query: str = Query(..., min_length=1, description="Text search query"),
    k: int = Query(12, ge=1, le=30, description="Number of results"),
):
    results = search_engine.search(query, k=k)

    payload = []
    for item in results:
        image_path = item.get("image_path", "")
        abs_path = normalize_image_path(image_path)
        if not abs_path.exists():
            continue

        try:
            rel_path = abs_path.relative_to(ROOT_DIR).as_posix()
        except ValueError:
            continue

        payload.append(
            {
                "score": round(float(item.get("score", 0.0)), 4),
                "class_name": item.get("class_name", ""),
                "image_url": f"/assets/{rel_path}",
            }
        )

    return {"query": query, "count": len(payload), "results": payload}


app.mount("/assets", StaticFiles(directory=ROOT_DIR), name="assets")
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")
