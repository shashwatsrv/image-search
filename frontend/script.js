const EXAMPLE_PROMPTS = [
  "golden hour over mountains",
  "rainy street at night",
  "minimalist home office",
  "foggy forest in the morning",
  "crowded city market",
  "lone tree in a field",
  "astronaut floating in space",
  "vintage car on a desert road",
];

const form = document.getElementById("search-form");
const input = document.getElementById("query-input");
const kSelect = document.getElementById("k-select");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const loadMoreBtn = document.getElementById("load-more-btn");
const surpriseBtn = document.getElementById("surprise-btn");
const themeToggleBtn = document.getElementById("theme-toggle");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxClose = document.getElementById("lightbox-close");

let allResults = [];
let visibleCount = 0;
const BATCH_SIZE = 8;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function toAbsolute(url) {
  return new URL(url, window.location.origin).toString();
}

function renderResults() {
  const items = allResults.slice(0, visibleCount);
  if (!items.length) {
    resultsEl.innerHTML = "";
    loadMoreBtn.classList.add("hidden");
    statusEl.textContent = "No results found.";
    return;
  }

  resultsEl.innerHTML = items
    .map((item) => {
      const safeClass = escapeHtml(item.class_name || "unknown");
      const safeImage = escapeHtml(item.image_url);
      const score = Number(item.score || 0);
      return `
      <article class="card">
        <button class="thumb-btn" type="button" data-lightbox="${safeImage}">
          <img src="${safeImage}" alt="${safeClass}" loading="lazy" />
        </button>
        <div class="meta">
          <span>${safeClass}</span>
          <span class="score">${(score * 100).toFixed(1)}%</span>
        </div>
        <div class="card-actions">
          <a class="mini-action" href="${safeImage}" download target="_blank" rel="noreferrer">Download</a>
          <button class="mini-action" type="button" data-copy="${safeImage}">Copy link</button>
        </div>
      </article>
    `;
    })
    .join("");

  loadMoreBtn.classList.toggle("hidden", visibleCount >= allResults.length);
}

function applyTheme(theme) {
  document.body.setAttribute("data-theme", theme);
  themeToggleBtn.textContent = theme === "dark" ? "Switch to Light" : "Switch to Dark";
  localStorage.setItem("image-search-theme", theme);
}

async function runSearch(query, k) {
  statusEl.textContent = `Searching for "${query}"...`;
  resultsEl.innerHTML = "";
  loadMoreBtn.classList.add("hidden");

  const start = performance.now();
  const params = new URLSearchParams({ query, k: String(k) });
  const response = await fetch(`/api/search?${params.toString()}`);
  const elapsed = ((performance.now() - start) / 1000).toFixed(2);

  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`);
  }

  const data = await response.json();
  allResults = Array.isArray(data.results) ? data.results : [];
  visibleCount = Math.min(BATCH_SIZE, allResults.length);
  statusEl.textContent = `Found ${data.count} result(s) in ${elapsed}s for "${data.query}".`;
  renderResults();
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  const k = Number(kSelect.value);
  if (!query) {
    statusEl.textContent = "Please enter a search query.";
    return;
  }

  try {
    await runSearch(query, k);
  } catch (error) {
    statusEl.textContent = `Search failed: ${error.message}`;
  }
});

loadMoreBtn.addEventListener("click", () => {
  visibleCount = Math.min(visibleCount + BATCH_SIZE, allResults.length);
  renderResults();
});

surpriseBtn.addEventListener("click", async () => {
  const randomPrompt = EXAMPLE_PROMPTS[Math.floor(Math.random() * EXAMPLE_PROMPTS.length)];
  input.value = randomPrompt;
  try {
    await runSearch(randomPrompt, Number(kSelect.value));
  } catch (error) {
    statusEl.textContent = `Search failed: ${error.message}`;
  }
});

themeToggleBtn.addEventListener("click", () => {
  const current = document.body.getAttribute("data-theme") || "dark";
  applyTheme(current === "dark" ? "light" : "dark");
});

resultsEl.addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }

  const copyUrl = target.getAttribute("data-copy");
  if (copyUrl) {
    try {
      await navigator.clipboard.writeText(toAbsolute(copyUrl));
      statusEl.textContent = "Image link copied to clipboard.";
    } catch {
      statusEl.textContent = "Could not copy link.";
    }
    return;
  }

  const lightboxTrigger = target.closest("[data-lightbox]");
  if (lightboxTrigger instanceof HTMLElement) {
    const src = lightboxTrigger.getAttribute("data-lightbox");
    if (src) {
      lightboxImage.src = src;
      lightbox.showModal();
    }
  }
});

lightboxClose.addEventListener("click", () => {
  lightbox.close();
});

lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    lightbox.close();
  }
});

const savedTheme = localStorage.getItem("image-search-theme");
applyTheme(savedTheme === "light" ? "light" : "dark");
