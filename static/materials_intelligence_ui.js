(function () {
  const searchEndpoint = document.body.dataset.searchEndpoint;

  const elements = {
    form:          document.getElementById("searchForm"),
    imageInput:    document.getElementById("imageInput"),
    dropzone:      document.getElementById("dropzone"),
    previewImage:  document.getElementById("previewImage"),
    previewName:   document.getElementById("previewName"),
    previewHint:   document.getElementById("previewHint"),
    submitBtn:     document.getElementById("submitBtn"),
    statusBar:     document.getElementById("statusBar"),
    emptyState:    document.getElementById("emptyState"),
    resultCard:    document.getElementById("resultCard"),
    resultImage:   document.getElementById("resultImage"),
    resultDomain:  document.getElementById("resultDomain"),
    resultScore:   document.getElementById("resultScore"),
    resultReference: document.getElementById("resultReference"),
    resultName:    document.getElementById("resultName"),
    adnButton:     document.getElementById("adnButton"),
  };

  let previewUrl = null;

  // ── Helpers ────────────────────────────────────────────────────────────────

  function setStatus(message, tone) {
    elements.statusBar.textContent = message;
    elements.statusBar.classList.remove("is-error", "is-success", "is-warning");
    if (tone === "error")   elements.statusBar.classList.add("is-error");
    if (tone === "success") elements.statusBar.classList.add("is-success");
    if (tone === "warning") elements.statusBar.classList.add("is-warning");
  }

  function formatPct(value) {
    if (value == null || Number.isNaN(Number(value))) return "-";
    return `${Number(value).toFixed(2)}%`;
  }

  function revokePreview() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      previewUrl = null;
    }
  }

  function resetResult() {
    elements.resultCard.hidden = true;
    elements.emptyState.hidden = false;
    elements.emptyState.textContent = "Upload an image to see the closest reference and name.";
    elements.adnButton.setAttribute("href", "#");
  }

  function renderPreview(file) {
    revokePreview();
    if (!file) {
      elements.previewImage.removeAttribute("src");
      elements.previewName.textContent = "No image selected";
      elements.previewHint.textContent = "Your image will appear here before search.";
      return;
    }
    previewUrl = URL.createObjectURL(file);
    elements.previewImage.src = previewUrl;
    elements.previewName.textContent = file.name;
    elements.previewHint.textContent = `${Math.round(file.size / 1024)} KB`;
  }

  function renderMatch(match) {
    elements.resultImage.src     = match.image_url || "";
    elements.resultDomain.textContent = match.domain === "nuance" ? "Nuance" : "Matiere";
    elements.resultScore.textContent  = `Similarity ${formatPct(match.similarity_pct)}`;
    elements.resultReference.textContent = match.reference    || "-";
    elements.resultName.textContent      = match.display_name || match.material_name || "-";
    elements.adnButton.href = match.adn_url || "#";
    elements.resultCard.hidden  = false;
    elements.emptyState.hidden  = true;
  }

  // ── Main search ────────────────────────────────────────────────────────────

  async function submitSearch(event) {
    event.preventDefault();

    const file = elements.imageInput.files && elements.imageInput.files[0];
    if (!file) {
      setStatus("Select an image before searching.", "error");
      resetResult();
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("top_k", "1");

    elements.submitBtn.disabled = true;
    setStatus("Searching the closest match…", null);
    resetResult();

    try {
      const response = await fetch(searchEndpoint, {
        method: "POST",
        body: formData,
      });

      const payload = await response.json();

      // ── FIX: handle no_confident_match explicitly ──────────────────────────
      if (!response.ok || !payload.success) {
        if (payload.error === "no_confident_match") {
          // Show best score hint so the user understands why it was rejected
          const pct = payload.best_similarity_pct != null
            ? ` (best score: ${payload.best_similarity_pct.toFixed(1)}%,`
              + ` threshold: ${(payload.threshold * 100).toFixed(0)}%)`
            : "";
          setStatus(
            `No confident match found${pct}. This image is likely not in the database yet.`,
            "error"
          );
          // Show the empty state with a more specific message
          elements.emptyState.textContent =
            "No match above the confidence threshold. "
            + "The image may not be indexed yet.";
          elements.emptyState.hidden = false;
          return;
        }
        throw new Error(payload.error || payload.message || "Search failed");
      }

      const bestMatch = payload.results && payload.results[0];
      if (!bestMatch) {
        throw new Error("No match returned");
      }

      renderMatch(bestMatch);
      setStatus("Closest match found.", "success");

    } catch (error) {
      resetResult();
      setStatus(error.message || String(error), "error");
    } finally {
      elements.submitBtn.disabled = false;
    }
  }

  // ── Dropzone ───────────────────────────────────────────────────────────────

  function attachDropzoneHandlers() {
    const activeClass = "is-active";

    elements.dropzone.addEventListener("click", function () {
      elements.imageInput.click();
    });

    ["dragenter", "dragover"].forEach(function (eventName) {
      elements.dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        elements.dropzone.classList.add(activeClass);
      });
    });

    ["dragleave", "dragend"].forEach(function (eventName) {
      elements.dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        elements.dropzone.classList.remove(activeClass);
      });
    });

    elements.dropzone.addEventListener("drop", function (event) {
      event.preventDefault();
      elements.dropzone.classList.remove(activeClass);
      const files = event.dataTransfer && event.dataTransfer.files;
      if (!files || !files.length) return;
      elements.imageInput.files = files;
      renderPreview(files[0]);
      setStatus("Image loaded. Click Find Match.", null);
    });
  }

  // ── Init ───────────────────────────────────────────────────────────────────

  elements.imageInput.addEventListener("change", function () {
    const file = elements.imageInput.files && elements.imageInput.files[0];
    renderPreview(file);
    if (file) setStatus("Image loaded. Click Find Match.", null);
  });

  elements.form.addEventListener("submit", submitSearch);
  attachDropzoneHandlers();
  resetResult();
})();