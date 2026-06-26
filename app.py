import streamlit as st
import numpy as np
import faiss
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from models.embedder import embed_text, embed_image, embed_multimodal
from huggingface_hub import hf_hub_download
import random

EXAMPLE_PROMPTS = [
    "golden hour over mountains",
    "rainy street at night",
    "child laughing on a beach",
    "minimalist home office",
    "foggy forest in the morning",
    "crowded city market",
    "lone tree in a field",
    "close-up of coffee and a book",
    "astronaut floating in space",
    "vintage car on a desert road",
    "dog running through snow",
    "abstract architecture looking up",
    "candlelit dinner for two",
    "surfer catching a wave",
    "colorful hot air balloons",
]

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Image Search",
    page_icon="🔍",
    layout="wide",
)

# ── load once ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index…")
def load_data():
    from huggingface_hub import hf_hub_download
    repo = "shashwatsrv/unsplash-search-index"

    emb_path = hf_hub_download(repo, "embeddings_25k.npy",   repo_type="dataset")
    ids_path = hf_hub_download(repo, "ids_25k.npy",          repo_type="dataset")
    idx_path = hf_hub_download(repo, "index_hnsw_25k.faiss", repo_type="dataset")
    csv_path = hf_hub_download(repo, "photos.csv",           repo_type="dataset")

    embeddings = np.load(emb_path)
    ids        = np.load(ids_path, allow_pickle=True)
    index      = faiss.read_index(idx_path)

    df = pd.read_csv(csv_path, sep="\t", on_bad_lines="skip")
    df = df.dropna(subset=["photo_image_url"])
    df = df[df["photo_image_url"].str.startswith("https")]
    df = df.reset_index(drop=True)

    keep = ["photo_image_url", "photographer_username",
            "photo_description", "photo_alt_description"]
    keep = [c for c in keep if c in df.columns]
    meta = df[["photo_id"] + keep].set_index("photo_id").to_dict("index")
    return embeddings, ids, index, meta

embeddings, ids, index, meta = load_data()

PLACEHOLDER = Image.new("RGB", (256, 256), color=(220, 220, 220))

# ── helpers ───────────────────────────────────────────────────────────────────
def clean(val):
    s = str(val).strip() if val is not None else ""
    return "" if s.lower() == "nan" else s

@lru_cache(maxsize=2000)
def fetch_image(url: str | None) -> Image.Image:
    if not url:
        return PLACEHOLDER
    try:
        fetch_url = (url + "&w=400") if "?" in url else (url + "?w=400")
        res = requests.get(fetch_url, timeout=6)
        res.raise_for_status()
        img = Image.open(BytesIO(res.content)).convert("RGB")
        img.thumbnail((400, 400))
        return img
    except Exception:
        return PLACEHOLDER

def fetch_images_parallel(urls: list[str | None]) -> list[Image.Image]:
    with ThreadPoolExecutor(max_workers=10) as ex:
        return list(ex.map(fetch_image, urls))

# ── search ────────────────────────────────────────────────────────────────────
def search_vector(vec: np.ndarray, k: int = 9):
    """Core search — accepts any pre-computed embedding vector."""
    vec = vec.reshape(1, -1).astype("float32")
    distances, indices = index.search(vec, k)
    sims   = 1 - (distances[0] ** 2) / 2
    scores = (sims + 1) / 2
    photo_ids = [ids[i] for i in indices[0]]
    return photo_ids, scores

def search_text(query: str, k: int = 9):
    return search_vector(embed_text(query), k)

def search_img(img: Image.Image, k: int = 9):
    return search_vector(embed_image(img), k)

def search_combined(img: Image.Image, text: str, alpha: float, k: int = 9):
    return search_vector(embed_multimodal(image=img, text=text, alpha=alpha), k)

# ── session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "example_prompt" not in st.session_state:
    st.session_state.example_prompt = random.choice(EXAMPLE_PROMPTS)
if "results" not in st.session_state:
    st.session_state.results = None   # (photo_ids, scores) or None

def run_search(q: str):
    q = q.strip()
    if not q:
        return
    st.session_state.query = q
    if q not in st.session_state.history:
        st.session_state.history.insert(0, q)
        st.session_state.history = st.session_state.history[:8]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🔍 Semantic Image Search")
st.caption("Powered by CLIP embeddings + FAISS HNSW · Unsplash Lite dataset")

k = st.slider("Results to show", 1, 20, 9)

tab_text, tab_image, tab_combined = st.tabs(["🔤 Text", "🖼️ Image", "🔤 + 🖼️ Combined"])

# ── Tab 1: Text ───────────────────────────────────────────────────────────────
with tab_text:
    with st.form(key="search_form", border=False):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            typed = st.text_input(
                "Query",
                placeholder=f"e.g. {st.session_state.example_prompt}",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button(
                "Search", use_container_width=True, type="primary"
            )

    if submitted:
        effective_query = typed.strip() or st.session_state.example_prompt
        run_search(effective_query)
        with st.spinner(f'Searching for **"{effective_query}"**…'):
            st.session_state.results = search_text(effective_query, k)

    # recent-search chips
    if st.session_state.history:
        st.write("**Recent:**")
        chip_cols = st.columns(len(st.session_state.history))
        for i, past in enumerate(st.session_state.history):
            if chip_cols[i].button(past, key=f"chip_{i}"):
                run_search(past)
                with st.spinner(f'Searching for **"{past}"**…'):
                    st.session_state.results = search_text(past, k)
                st.rerun()

# ── Tab 2: Image ──────────────────────────────────────────────────────────────
with tab_image:
    uploaded = st.file_uploader(
        "Upload an image to find visually similar photos",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded:
        query_img = Image.open(uploaded).convert("RGB")
        st.image(query_img, width=280, caption="Query image")
        if st.button("Find similar images", type="primary"):
            with st.spinner("Embedding image and searching…"):
                st.session_state.results = search_img(query_img, k)
            st.session_state.query = f"[image: {uploaded.name}]"

# ── Tab 3: Combined ───────────────────────────────────────────────────────────
with tab_combined:
    st.caption("Upload an image AND add a text description — CLIP blends both into one query.")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        combo_img = st.file_uploader(
            "Image", type=["jpg", "jpeg", "png", "webp"], key="combo_img"
        )
        if combo_img:
            st.image(Image.open(combo_img), width=220)
    with col_b:
        combo_text = st.text_input("Text refinement", placeholder="e.g. at night, in winter…")
        alpha = st.slider(
            "Image ← weight → Text",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="0 = text only, 1 = image only, 0.7 = mostly image with text hint",
        )

    if st.button("Search combined", type="primary"):
        if not combo_img:
            st.warning("Upload an image first.")
        else:
            pil_img = Image.open(combo_img).convert("RGB")
            text_hint = combo_text.strip() or None
            with st.spinner("Blending embeddings and searching…"):
                st.session_state.results = search_combined(pil_img, text_hint, alpha, k)
            label = f"[image + '{text_hint}']" if text_hint else "[image]"
            st.session_state.query = label

# ── Results (shared across all tabs) ─────────────────────────────────────────
if st.session_state.results:
    photo_ids, scores = st.session_state.results
    urls   = [meta.get(pid, {}).get("photo_image_url") for pid in photo_ids]
    images = fetch_images_parallel(urls)

    st.divider()
    if st.session_state.query:
        st.markdown(f"**Results for:** {st.session_state.query}")

    n_cols  = min(4, k)
    columns = st.columns(n_cols)

    for i, (img, score, pid) in enumerate(zip(images, scores, photo_ids)):
        row_meta     = meta.get(pid, {})
        photographer = row_meta.get("photographer_username", "")
        description  = clean(row_meta.get("photo_description")) or clean(row_meta.get("photo_alt_description"))
        orig_url     = row_meta.get("photo_image_url", "")

        with columns[i % n_cols]:
            st.image(img, use_container_width=True)

            pct = score * 100
            bar_color = (
                "#2ecc71" if pct >= 75 else
                "#f39c12" if pct >= 50 else
                "#e74c3c"
            )
            st.markdown(
                f"""
                <div style="background:#e0e0e0;border-radius:4px;height:6px;margin:2px 0 4px">
                  <div style="width:{pct:.1f}%;background:{bar_color};
                              height:6px;border-radius:4px"></div>
                </div>
                <small style="color:#555">{pct:.1f}% match
                {"· @" + photographer if photographer else ""}</small>
                """,
                unsafe_allow_html=True,
            )

            if description:
                with st.expander("📄 Description", expanded=False):
                    st.write(description[:200] + ("…" if len(description) > 200 else ""))

            if orig_url:
                st.markdown(
                    f'<a href="{orig_url}" target="_blank" '
                    f'style="font-size:0.75rem">🔗 View on Unsplash</a>',
                    unsafe_allow_html=True,
                )

            st.divider()

else:
    st.info("Enter a query above to search 25 000 Unsplash photos by meaning, not keywords.")