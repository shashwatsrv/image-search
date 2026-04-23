import streamlit as st
import numpy as np
import faiss
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from models.embedder import embed_text
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
    embeddings = np.load("data/unsplash/embeddings_25k.npy")
    ids        = np.load("data/unsplash/ids_25k.npy", allow_pickle=True)
    index      = faiss.read_index("data/unsplash/index_hnsw_25k.faiss")

    df = pd.read_csv("data/unsplash_lite/photos.csv", sep="\t", on_bad_lines="skip")
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
def search_text(query: str, k: int = 5):
    vec = embed_text(query).reshape(1, -1).astype("float32")
    distances, indices = index.search(vec, k)
    sims   = 1 - (distances[0] ** 2) / 2
    scores = (sims + 1) / 2
    photo_ids = [ids[i] for i in indices[0]]
    return photo_ids, scores

# ── session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "query" not in st.session_state:
    st.session_state.query = ""
if "example_prompt" not in st.session_state:
    st.session_state.example_prompt = random.choice(EXAMPLE_PROMPTS)

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

# st.form intercepts the Enter key reliably
with st.form(key="search_form", border=False):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        typed = st.text_input(
            "Query",
            value=st.session_state.query,
            placeholder=f"e.g. {st.session_state.example_prompt}",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button(
            "Search", use_container_width=True, type="primary"
        )

if submitted:
    # if the user typed nothing, fall back to the example prompt
    effective_query = typed.strip() or st.session_state.example_prompt
    run_search(effective_query)

k = st.slider("Results to show", 1, 20, 9)

# recent-search chips
if st.session_state.history:
    st.write("**Recent:**")
    chip_cols = st.columns(len(st.session_state.history))
    for i, past in enumerate(st.session_state.history):
        if chip_cols[i].button(past, key=f"chip_{i}"):
            run_search(past)
            st.rerun()

active_query = st.session_state.query

# ── results ───────────────────────────────────────────────────────────────────
if active_query:
    with st.spinner(f'Searching for **"{active_query}"**…'):
        photo_ids, scores = search_text(active_query, k)

    urls   = [meta.get(pid, {}).get("photo_image_url") for pid in photo_ids]
    images = fetch_images_parallel(urls)

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