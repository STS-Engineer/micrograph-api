(function () {
  const searchEndpoint = document.body.dataset.searchEndpoint;

  const elements = {
    form: document.getElementById("searchForm"),
    imageInput: document.getElementById("imageInput"),
    dropzone: document.getElementById("dropzone"),
    previewImage: document.getElementById("previewImage"),
    previewName: document.getElementById("previewName"),
    previewHint: document.getElementById("previewHint"),
    topK: document.getElementById("topK"),
    topKValue: document.getElementById("topKValue"),
    submitBtn: document.getElementById("submitBtn"),
    statusBar: document.getElementById("statusBar"),
    metricCount: document.getElementById("metricCount"),
    metricBest: document.getElementById("metricBest"),
    metricAverage: document.getElementById("metricAverage"),
    metricSignal: document.getElementById("metricSignal"),
    searchNarrative: document.getElementById("searchNarrative"),
    querySnapshot: document.getElementById("querySnapshot"),
    resultsList: document.getElementById("resultsList"),
    detailsContent: document.getElementById("detailsContent"),
  };

  const state = {
    previewUrl: null,
    results: [],
    selectedMatiereId: null,
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function maybeParseJson(value) {
    if (typeof value !== "string") {
      return value;
    }
    try {
      return JSON.parse(value);
    } catch (_err) {
      return value;
    }
  }

  function formatPct(value) {
    if (value == null || Number.isNaN(Number(value))) {
      return "-";
    }
    return `${Number(value).toFixed(2)}%`;
  }

  function formatSimilarity(score) {
    if (score == null || Number.isNaN(Number(score))) {
      return "-";
    }
    return Number(score).toFixed(4);
  }

  function describeSignal(score) {
    if (score == null) {
      return "No score";
    }
    if (score >= 0.97) {
      return "Very strong match";
    }
    if (score >= 0.94) {
      return "Strong candidate";
    }
    if (score >= 0.9) {
      return "Relevant candidate";
    }
    return "Review manually";
  }

  function buildNarrative(summary, results) {
    if (!results.length) {
      return "No materials were returned for this query image.";
    }

    const best = summary.best_match;
    const gap = summary.confidence_gap;
    if (!best) {
      return "The search completed, but there is no best-match signal yet.";
    }

    const lead = `${best.material_name || "Top result"} (${best.reference || "no reference"}) leads at ${formatPct(best.similarity_pct)}.`;
    if (gap == null) {
      return `${lead} There is no second candidate to compare against.`;
    }
    if (gap >= 0.03) {
      return `${lead} The lead over the second candidate is wide, which usually indicates a stable proposal set.`;
    }
    if (gap >= 0.012) {
      return `${lead} The first candidate is ahead, but the next result is still worth checking in the detail panel.`;
    }
    return `${lead} The top scores are clustered together, so compare the first materials carefully before selecting one.`;
  }

  function setStatus(message, tone) {
    elements.statusBar.textContent = message;
    elements.statusBar.classList.remove("is-error", "is-success");
    if (tone === "error") {
      elements.statusBar.classList.add("is-error");
    } else if (tone === "success") {
      elements.statusBar.classList.add("is-success");
    }
  }

  function resetMetrics() {
    elements.metricCount.textContent = "0";
    elements.metricBest.textContent = "-";
    elements.metricAverage.textContent = "-";
    elements.metricSignal.textContent = "-";
  }

  function updateTopKLabel() {
    elements.topKValue.textContent = elements.topK.value;
  }

  function revokePreviewUrl() {
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
      state.previewUrl = null;
    }
  }

  function renderPreview(file) {
    revokePreviewUrl();

    if (!file) {
      elements.previewImage.removeAttribute("src");
      elements.previewName.textContent = "No file selected";
      elements.previewHint.textContent = "The uploaded image stays in-browser until you run the search.";
      return;
    }

    state.previewUrl = URL.createObjectURL(file);
    elements.previewImage.src = state.previewUrl;
    elements.previewName.textContent = file.name;
    elements.previewHint.textContent = `${Math.round(file.size / 1024)} KB`;
  }

  function renderQuerySnapshot(payload) {
    const best = payload.summary.best_match;
    const imageUrl = payload.query.uploaded_image_url;

    if (!imageUrl) {
      elements.querySnapshot.className = "query-snapshot query-snapshot-empty";
      elements.querySnapshot.innerHTML = `
        <div class="query-copy">
          <p class="query-label">Query snapshot</p>
          <p class="query-title">Search executed</p>
          <p class="query-text">${escapeHtml(payload.query.filename || "Uploaded image processed.")}</p>
        </div>
      `;
      return;
    }

    elements.querySnapshot.className = "query-snapshot";
    elements.querySnapshot.innerHTML = `
      <img src="${escapeHtml(imageUrl)}" alt="Uploaded query image" />
      <div class="query-copy">
        <p class="query-label">Query snapshot</p>
        <p class="query-title">${escapeHtml(payload.query.filename || "Uploaded image")}</p>
        <p class="query-text">
          ${best ? `Best match: <strong>${escapeHtml(best.material_name || "")}</strong> (${escapeHtml(best.reference || "")}) at ${formatPct(best.similarity_pct)}.` : "No ranked material found."}
        </p>
      </div>
    `;
  }

  function renderMetrics(payload) {
    const summary = payload.summary;
    elements.metricCount.textContent = String(summary.result_count || 0);
    elements.metricBest.textContent = summary.best_match ? formatPct(summary.best_match.similarity_pct) : "-";
    elements.metricAverage.textContent = summary.average_similarity_pct != null ? formatPct(summary.average_similarity_pct) : "-";
    elements.metricSignal.textContent = summary.confidence_gap != null ? summary.confidence_gap.toFixed(4) : "-";
    elements.searchNarrative.textContent = buildNarrative(summary, payload.results);
  }

  function renderResults(results) {
    state.results = results.slice();

    if (!results.length) {
      elements.resultsList.innerHTML = `
        <div class="details-block">
          <h3 class="detail-block-title">No materials found</h3>
          <p class="detail-subtitle">Try a different crop, a sharper image, or a smaller top-K range.</p>
        </div>
      `;
      return;
    }

    elements.resultsList.innerHTML = results.map((result) => `
      <button class="result-card" type="button" data-matiere-id="${escapeHtml(result.matiere_id)}">
        <div class="result-rank">#${escapeHtml(result.rank)}</div>
        <div class="result-thumb">
          <img src="${escapeHtml(result.image_url || "")}" alt="${escapeHtml(result.material_name || "Result image")}" />
        </div>
        <div class="result-main">
          <div class="result-title">${escapeHtml(result.material_name || "Unknown material")}</div>
          <div class="result-meta">${escapeHtml(result.reference || "No reference")} | ${escapeHtml(result.type_matiere || "Type n/a")}</div>
          <div class="score-row">
            <div class="score-bar"><span style="width:${Math.max(0, Math.min(100, Number(result.similarity_pct || 0)))}%"></span></div>
            <span class="score-label">${formatPct(result.similarity_pct)}</span>
          </div>
          <span class="chip result-signal">${escapeHtml(describeSignal(result.similarity))}</span>
        </div>
      </button>
    `).join("");
  }

  function renderDetailsLoading(result) {
    elements.detailsContent.className = "details-content";
    elements.detailsContent.innerHTML = `
      <div class="details-block loading-pulse">
        <h3 class="detail-block-title">${escapeHtml(result.material_name || "Loading material")}</h3>
        <p class="detail-subtitle">Fetching database detail for ${escapeHtml(result.reference || "selected result")}.</p>
      </div>
    `;
  }

  function renderEmptyDetails(message) {
    elements.detailsContent.className = "details-content empty-state";
    elements.detailsContent.textContent = message;
  }

  function renderDetails(detailPayload, result) {
    const material = detailPayload.material || {};
    const fiches = Array.isArray(detailPayload.fiches_matieres) ? detailPayload.fiches_matieres : [];
    const specifications = Array.isArray(detailPayload.specifications) ? detailPayload.specifications : [];
    const expertNotes = Array.isArray(detailPayload.expert_notes) ? detailPayload.expert_notes : [];
    const summary = detailPayload.summary || {};

    const fieldPairs = [
      ["Material ID", material.matiere_id],
      ["Name", material.nom_matiere],
      ["Reference", material.reference],
      ["Type", material.type_matiere],
    ].filter(([, value]) => value !== null && value !== undefined && value !== "");

    const ficheMarkup = fiches.slice(0, 4).map((fiche) => `
      <div class="detail-list-item">
        <strong>Fiche #${escapeHtml(fiche.fiche_id)}</strong>
        <p>Created: ${escapeHtml(fiche.date_creation_fiche || "n/a")}</p>
        <p>Updated: ${escapeHtml(fiche.derniere_modification || "n/a")}</p>
      </div>
    `).join("");

    const specMarkup = specifications.slice(0, 3).map((spec) => {
      const payload = maybeParseJson(spec.donnees);
      return `
        <details>
          <summary>Specification #${escapeHtml(spec.spec_id)} | ${escapeHtml(spec.source_type || "unknown source")}</summary>
          <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
        </details>
      `;
    }).join("");

    const noteMarkup = expertNotes.slice(0, 2).map((note) => `
      <details>
        <summary>Expert note #${escapeHtml(note.id)} | ${escapeHtml(note.created_at || "n/a")}</summary>
        <pre>${escapeHtml(JSON.stringify(note.note_json || {}, null, 2))}</pre>
      </details>
    `).join("");

    elements.detailsContent.className = "details-content";
    elements.detailsContent.innerHTML = `
      <div class="details-block detail-hero">
        ${result.image_url ? `<img src="${escapeHtml(result.image_url)}" alt="${escapeHtml(result.material_name || "Selected result")}" />` : ""}
        <div class="detail-headline">
          <h3>${escapeHtml(result.material_name || "Selected material")}</h3>
          <p class="detail-subtitle">${escapeHtml(result.reference || "No reference")} | ${escapeHtml(result.type_matiere || "Type n/a")} | similarity ${formatSimilarity(result.similarity)}</p>
          <div class="detail-chip-row">
            <span class="chip">${escapeHtml(describeSignal(result.similarity))}</span>
            <span class="chip">Rank #${escapeHtml(result.rank)}</span>
          </div>
        </div>
      </div>

      <div class="details-block">
        <h3 class="detail-block-title">Database summary</h3>
        <div class="detail-stat-row">
          <div class="detail-stat"><span class="metric-label">Fiches</span><strong>${escapeHtml(summary.num_fiches ?? 0)}</strong></div>
          <div class="detail-stat"><span class="metric-label">Specifications</span><strong>${escapeHtml(summary.num_specifications ?? 0)}</strong></div>
          <div class="detail-stat"><span class="metric-label">Expert notes</span><strong>${escapeHtml(summary.num_expert_notes ?? 0)}</strong></div>
        </div>
      </div>

      <div class="details-block">
        <h3 class="detail-block-title">Material fields</h3>
        <div class="detail-field-grid">
          ${fieldPairs.map(([label, value]) => `
            <div class="detail-field">
              <span class="metric-label">${escapeHtml(label)}</span>
              <strong>${escapeHtml(value)}</strong>
            </div>
          `).join("")}
        </div>
      </div>

      <div class="details-block">
        <h3 class="detail-block-title">Recent fiches</h3>
        <div class="detail-list">
          ${ficheMarkup || '<div class="detail-list-item"><strong>No fiche rows</strong><p>No fiche data was returned for this material.</p></div>'}
        </div>
      </div>

      <div class="details-block">
        <h3 class="detail-block-title">Specification samples</h3>
        <div class="detail-list">
          ${specMarkup || '<div class="detail-list-item"><strong>No specification rows</strong><p>No specification payloads were returned for this material.</p></div>'}
        </div>
      </div>

      <div class="details-block">
        <h3 class="detail-block-title">Expert note samples</h3>
        <div class="detail-list">
          ${noteMarkup || '<div class="detail-list-item"><strong>No expert notes</strong><p>No expert note payloads were returned for this material.</p></div>'}
        </div>
      </div>
    `;
  }

  async function loadDetails(matiereId) {
    const result = state.results.find((item) => String(item.matiere_id) === String(matiereId));
    if (!result) {
      return;
    }

    state.selectedMatiereId = result.matiere_id;
    [...elements.resultsList.querySelectorAll(".result-card")].forEach((card) => {
      card.classList.toggle("is-selected", card.dataset.matiereId === String(result.matiere_id));
    });
    renderDetailsLoading(result);

    try {
      const response = await fetch(result.detail_url, { headers: { Accept: "application/json" } });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || data.message || "Unable to load material details");
      }
      renderDetails(data, result);
    } catch (error) {
      renderEmptyDetails(error.message || String(error));
    }
  }

  async function submitSearch(event) {
    event.preventDefault();

    const file = elements.imageInput.files && elements.imageInput.files[0];
    if (!file) {
      setStatus("Select a micrograph image before searching.", "error");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("top_k", elements.topK.value);

    elements.submitBtn.disabled = true;
    setStatus("Running similarity search against the material base...", "info");
    renderEmptyDetails("A material detail panel will open automatically after the search.");
    elements.resultsList.innerHTML = "";
    resetMetrics();

    try {
      const response = await fetch(searchEndpoint, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok || !payload.success) {
        throw new Error(payload.error || payload.message || "Search failed");
      }

      renderQuerySnapshot(payload);
      renderMetrics(payload);
      renderResults(payload.results || []);
      setStatus(`Search complete. ${payload.summary.result_count} proposed material(s) returned.`, "success");

      const firstResult = payload.results && payload.results[0];
      if (firstResult) {
        await loadDetails(firstResult.matiere_id);
      } else {
        renderEmptyDetails("The search completed, but no material candidate was returned.");
      }
    } catch (error) {
      setStatus(error.message || String(error), "error");
      renderEmptyDetails("Search failed. Review the message above, then try again.");
    } finally {
      elements.submitBtn.disabled = false;
    }
  }

  function attachDropzoneHandlers() {
    const activeClass = "is-active";

    elements.dropzone.addEventListener("click", function () {
      elements.imageInput.click();
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      elements.dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        elements.dropzone.classList.add(activeClass);
      });
    });

    ["dragleave", "dragend"].forEach((eventName) => {
      elements.dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        elements.dropzone.classList.remove(activeClass);
      });
    });

    elements.dropzone.addEventListener("drop", function (event) {
      event.preventDefault();
      elements.dropzone.classList.remove(activeClass);
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) {
        return;
      }
      elements.imageInput.files = files;
      renderPreview(files[0]);
      setStatus("Image loaded. Ready to search.", "info");
    });
  }

  function attachResultSelectionHandler() {
    elements.resultsList.addEventListener("click", function (event) {
      const card = event.target.closest(".result-card");
      if (!card) {
        return;
      }
      loadDetails(card.dataset.matiereId);
    });
  }

  elements.topK.addEventListener("input", updateTopKLabel);
  elements.imageInput.addEventListener("change", function () {
    const file = elements.imageInput.files && elements.imageInput.files[0];
    renderPreview(file);
    if (file) {
      setStatus("Image loaded. Ready to search.", "info");
    }
  });
  elements.form.addEventListener("submit", submitSearch);

  attachDropzoneHandlers();
  attachResultSelectionHandler();
  updateTopKLabel();
  resetMetrics();
})();
