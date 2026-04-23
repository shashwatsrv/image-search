const form = document.getElementById("search-form");
const input = document.getElementById("query-input");
const kSelect = document.getElementById("k-select");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

function renderResults(items) {
  if (!items.length) {
    resultsEl.innerHTML = "";
    statusEl.textContent = "No results found.";
    return;
  }

  resultsEl.innerHTML = items
    .map(
      (item) => `
      <article class="card">
        <img src="${item.image_url}" alt="${item.class_name || "search result"}" loading="lazy" />
        <div class="meta">
          <span>${item.class_name || "unknown"}</span>
          <span class="score">${(item.score * 100).toFixed(1)}%</span>
        </div>
      </article>
    `
    )
    .join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = input.value.trim();
  const k = Number(kSelect.value);

  if (!query) {
    statusEl.textContent = "Please enter a search query.";
    return;
  }

  statusEl.textContent = `Searching for "${query}"...`;
  resultsEl.innerHTML = "";

  try {
    const params = new URLSearchParams({ query, k: String(k) });
    const response = await fetch(`/api/search?${params.toString()}`);

    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`);
    }

    const data = await response.json();
    statusEl.textContent = `Found ${data.count} result(s) for "${data.query}".`;
    renderResults(data.results || []);
  } catch (error) {
    statusEl.textContent = `Search failed: ${error.message}`;
  }
});
