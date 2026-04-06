/**
 * Micrograph Match — UI v2 + Compare ADN
 *
 * Two views:
 *   1. Image Search  — POST /api/materials/browser-search  → single best match
 *   2. Compare ADN   — POST /api/materials/compare         → two searches + Groq diff
 *
 * Compare flow:
 *   a. Search image A via DINOv2 + pgvector  → match A
 *   b. Search image B via DINOv2 + pgvector  → match B
 *   c. Backend computes cosine similarity between the two embeddings
 *   d. Groq Llama generates a structured differential ADN report
 *   e. Frontend renders it in tabs: Differences / ADN-A / ADN-B
 */

(function () {
  "use strict";

  // ── Endpoints ────────────────────────────────────────────────────────────
  const SEARCH_ENDPOINT  = document.body.dataset.searchEndpoint;
  const COMPARE_ENDPOINT = document.body.dataset.compareEndpoint;

  // ── Loader copy ──────────────────────────────────────────────────────────
  const SEARCH_STEPS = [
    "Opening image & converting to RGB…",
    "Generating multi-scale crop views…",
    "Extracting DINOv2-large CLS embeddings…",
    "L2-normalising & averaging vectors…",
    "Running pgvector cosine search…",
  ];
  const COMPARE_STEPS = [
    "Searching image A via DINOv2…",
    "Searching image B via DINOv2…",
    "Computing cross-embedding cosine similarity…",
    "Retrieving ADN data for both matches…",
    "Generating differential report with Groq Llama…",
  ];

  // ══════════════════════════════════════════════════════════════════════════
  //  SHARED HELPERS
  // ══════════════════════════════════════════════════════════════════════════

  function formatPct(v) {
    if (v == null || Number.isNaN(Number(v))) return "—";
    return Number(v).toFixed(1) + "%";
  }

  function formatBytes(b) {
    if (!b) return "—";
    if (b < 1024) return b + " B";
    if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
    return (b / 1048576).toFixed(2) + " MB";
  }

  function simPctFromMatch(m) {
    if (!m) return null;
    if (m.similarity_pct != null) return m.similarity_pct;
    if (m.similarity    != null) return m.similarity * 100;
    return null;
  }

  function escapeHtml(text) {
    return String(text == null ? "" : text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function labelForDomain(domain) {
    const normalized = String(domain || "").toLowerCase();
    if (normalized === "nuance") return "Nuance";
    if (normalized === "test") return "Match";
    return "Matière";
  }

  function chipClassForDomain(domain) {
    const normalized = String(domain || "").toLowerCase();
    if (normalized === "nuance") return " is-nuance";
    if (normalized === "test") return "";
    return " is-matiere";
  }

  /** Very lightweight Markdown → HTML (enough for ADN content) */
  function markdownToHtml(md) {
    if (!md) return "";
    const codeBlocks = [];
    let source = String(md).replace(/\r\n?/g, "\n");
    source = source.replace(/```([a-zA-Z0-9_-]+)?\n([\s\S]*?)```/g, (_, lang, code) => {
      const token = `@@CODEBLOCK_${codeBlocks.length}@@`;
      const classAttr = lang ? ` class="language-${escapeHtml(lang)}"` : "";
      codeBlocks.push(`<pre><code${classAttr}>${escapeHtml(String(code || "").trim())}</code></pre>`);
      return token;
    });

    let html = source
      .replace(/^### (.+)$/gm, "<h3>$1</h3>")
      .replace(/^## (.+)$/gm,  "<h2>$1</h2>")
      .replace(/^# (.+)$/gm,   "<h1>$1</h1>")
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g,     "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/^---$/gm, "<hr>")
      .replace(/^[\-\*] (.+)$/gm, "<li>$1</li>")
      .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/\n/g, "<br>");

    html = html.replace(/(<li>[\s\S]+?<\/li>)+/g, m => "<ul>" + m + "</ul>");
    html = "<p>" + html + "</p>";
    html = html.replace(/<p>\s*<\/p>/g, "");
    html = html.replace(/<p>(<h[123]>)/g, "$1").replace(/(<\/h[123]>)<\/p>/g, "$1");
    html = html.replace(/<p>(<ul>)/g,     "$1").replace(/(<\/ul>)<\/p>/g,     "$1");
    html = html.replace(/<p>(<hr>)<\/p>/g, "$1");
    html = html.replace(/<p>(@@CODEBLOCK_\d+@@)<\/p>/g, "$1");
    html = html.replace(/@@CODEBLOCK_(\d+)@@/g, (_, idx) => codeBlocks[Number(idx)] || "");
    return html;
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  VIEW ROUTING
  // ══════════════════════════════════════════════════════════════════════════

  const views = document.querySelectorAll(".view");
  const navItems = document.querySelectorAll(".nav-item[data-view]");

  function switchView(viewId) {
    views.forEach(v => { v.hidden = (v.id !== "view-" + viewId); });
    navItems.forEach(n => {
      n.classList.toggle("is-active", n.dataset.view === viewId);
    });
  }

  navItems.forEach(n => {
    n.addEventListener("click", e => {
      e.preventDefault();
      switchView(n.dataset.view);
    });
  });

  // ══════════════════════════════════════════════════════════════════════════
  //  VIEW 1 — IMAGE SEARCH
  // ══════════════════════════════════════════════════════════════════════════

  const S = {
    form:        document.getElementById("searchForm"),
    imageInput:  document.getElementById("imageInput"),
    dropzone:    document.getElementById("dropzone"),
    previewWrap: document.getElementById("previewWrap"),
    previewImage:document.getElementById("previewImage"),
    previewName: document.getElementById("previewName"),
    previewSize: document.getElementById("previewSize"),
    clearBtn:    document.getElementById("clearBtn"),
    submitBtn:   document.getElementById("submitBtn"),
    statusBar:   document.getElementById("statusBar"),
    statusText:  document.getElementById("statusText"),
    emptyState:  document.getElementById("emptyState"),
    loaderState: document.getElementById("loaderState"),
    loaderSub:   document.getElementById("loaderSub"),
    resultCard:  document.getElementById("resultCard"),
    resultDomain:document.getElementById("resultDomain"),
    resultScore: document.getElementById("resultScore"),
    scoreBar:    document.getElementById("scoreBar"),
    resultRef:   document.getElementById("resultReference"),
    resultName:  document.getElementById("resultName"),
    adnBtn:      document.getElementById("adnButton"),
    cancelAdn:   document.getElementById("cancelAdnButton"),
    adnPanel:    document.getElementById("adnPanel"),
    adnMeta:     document.getElementById("adnMeta"),
    adnContent:  document.getElementById("adnContent"),
    dlPanel:     document.getElementById("downloadPanel"),
    dlLink:      document.getElementById("downloadLink"),
  };

  let sPreviewUrl = null;
  let sAdnCtrl    = null;
  let sLoopId     = null;
  let sLoadedAdnUrl = null;

  function sSetStatus(msg, tone) {
    S.statusText.textContent = msg;
    S.statusBar.classList.remove("is-success","is-error","is-warning","is-loading");
    if (tone) S.statusBar.classList.add("is-" + tone);
  }

  function sShowPanel(name) {
    S.emptyState.hidden  = name !== "empty";
    S.loaderState.hidden = name !== "loader";
    S.resultCard.hidden  = name !== "result";
  }

  function sRevokePreview() {
    if (sPreviewUrl) { URL.revokeObjectURL(sPreviewUrl); sPreviewUrl = null; }
  }

  function sRenderPreview(file) {
    sRevokePreview();
    if (!file) {
      S.previewWrap.hidden = true;
      S.dropzone.hidden    = false;
      S.submitBtn.disabled = true;
      return;
    }
    sPreviewUrl = URL.createObjectURL(file);
    S.previewImage.src = sPreviewUrl;
    S.previewName.textContent = file.name;
    S.previewSize.textContent = formatBytes(file.size);
    S.previewWrap.hidden = false;
    S.dropzone.hidden    = true;
    S.submitBtn.disabled = false;
  }

  function sClearPreview() {
    S.imageInput.value = "";
    sRenderPreview(null);
    sResetResult();
    sSetStatus("Waiting for an image.", null);
  }

  function sResetResult() {
    sShowPanel("empty");
    S.adnBtn.setAttribute("href","#");
    S.adnBtn.style.pointerEvents = "";
    S.adnBtn.style.opacity = "";
    S.adnBtn.removeAttribute("aria-disabled");
    S.adnBtn.removeAttribute("title");
    S.dlPanel.hidden = true;
    S.dlLink.setAttribute("href","#");
    S.cancelAdn.hidden = true;
    S.adnPanel.hidden = true;
    S.adnMeta.textContent = "-";
    S.adnContent.innerHTML = "";
    S.scoreBar.style.width = "0%";
    sLoadedAdnUrl = null;
  }

  function sRenderMatch(m) {
    const domain = (m.domain || "").toLowerCase();
    S.resultDomain.textContent = labelForDomain(domain);
    S.resultDomain.className = "domain-chip" + chipClassForDomain(domain);
    const pct = simPctFromMatch(m);
    S.resultScore.textContent = formatPct(pct);
    S.scoreBar.style.width = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      S.scoreBar.style.width = (pct != null ? Math.min(100, Math.max(0, pct)) : 0) + "%";
    }));
    S.resultRef.textContent  = m.reference || "—";
    S.resultName.textContent = m.display_name || m.material_name || "—";
    S.adnBtn.href = m.adn_url || "#";
    S.adnBtn.dataset.docxUrl = m.adn_docx_url || "";
    const hasAdn = !!m.adn_url;
    S.adnBtn.style.pointerEvents = hasAdn ? "" : "none";
    S.adnBtn.style.opacity = hasAdn ? "" : ".55";
    S.adnBtn.setAttribute("aria-disabled", hasAdn ? "false" : "true");
    S.adnBtn.removeAttribute("title");
    S.adnPanel.hidden = true;
    S.adnMeta.textContent = "-";
    S.adnContent.innerHTML = "";
    sLoadedAdnUrl = null;
    sShowPanel("result");
  }

  function sStartLoader() {
    let i = 0;
    S.loaderSub.textContent = SEARCH_STEPS[0];
    sLoopId = setInterval(() => {
      i = (i + 1) % SEARCH_STEPS.length;
      S.loaderSub.textContent = SEARCH_STEPS[i];
    }, 900);
  }
  function sStopLoader() { if (sLoopId) { clearInterval(sLoopId); sLoopId = null; } }

  async function sHandleAdnClick(e) {
    e.preventDefault();
    const url = S.adnBtn.getAttribute("href");
    if (!url || url === "#") return;
    sAdnCtrl = new AbortController();
    S.adnBtn.textContent = "Generating…";
    S.adnBtn.style.pointerEvents = "none";
    S.cancelAdn.hidden = false;
    S.adnBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 14l4-4 3 3 5-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Loading ADN...`;
    sSetStatus("Loading ADN for the closest match...","loading");
    sSetStatus("Generating ADN with Groq Llama…","loading");
    try {
      const r = await fetch(url, { headers:{Accept:"application/json"}, signal: sAdnCtrl.signal });
      const p = await r.json();
      if (!r.ok || !p.success || !p.absolute_url) throw new Error(p.error || "ADN generation failed");
      S.dlLink.href = p.absolute_url;
      S.dlPanel.hidden = false;
      window.open(p.absolute_url,"_blank","noopener,noreferrer");
      sSetStatus("ADN document generated.","success");
    } catch(err) {
      sSetStatus(err.name === "AbortError" ? "Cancelled." : (err.message || String(err)),
                 err.name === "AbortError" ? "warning" : "error");
    } finally {
      sAdnCtrl = null;
      S.adnBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 14l4-4 3 3 5-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Generate / See ADN`;
      S.adnBtn.style.pointerEvents = "";
      S.cancelAdn.hidden = true;
    }
  }

  async function sHandleInlineAdnClickLegacy(e) {
    e.preventDefault();
    const url = S.adnBtn.getAttribute("href");
    if (!url || url === "#") return;
    if (!S.adnPanel.hidden && sLoadedAdnUrl === url) {
      S.adnPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
      sSetStatus(`ADN ready for ${S.resultRef.textContent || "this match"}.`, "success");
      return;
    }
    sAdnCtrl = new AbortController();
    S.adnBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 14l4-4 3 3 5-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Loading ADNâ€¦`;
    S.adnBtn.style.pointerEvents = "none";
    S.cancelAdn.hidden = false;
    sSetStatus("Loading ADN for the closest matchâ€¦","loading");
    try {
      const r = await fetch(url, { headers:{Accept:"application/json"}, signal: sAdnCtrl.signal });
      const p = await r.json();
      if (!r.ok || !p.success || !p.adn_markdown) throw new Error(p.error || p.message || "ADN retrieval failed");
      S.adnContent.innerHTML = markdownToHtml(p.adn_markdown);
      S.adnMeta.textContent = `${labelForDomain(p.domain)} - ${p.reference || S.resultRef.textContent || "Reference unavailable"}`;
      S.adnMeta.textContent = `${labelForDomain(p.domain)} • ${p.reference || S.resultRef.textContent || "Reference unavailable"}`;
      S.adnPanel.hidden = false;
      S.dlPanel.hidden = true;
      sLoadedAdnUrl = url;
      S.adnPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      sSetStatus(`ADN loaded for ${p.reference || S.resultRef.textContent || "the closest match"}.`,"success");
    } catch(err) {
      if (err.name !== "AbortError") S.adnPanel.hidden = true;
      sSetStatus(err.name === "AbortError" ? "Cancelled." : (err.message || String(err)),
                 err.name === "AbortError" ? "warning" : "error");
    } finally {
      sAdnCtrl = null;
      S.adnBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 14l4-4 3 3 5-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Generate / See ADN`;
      S.adnBtn.style.pointerEvents = "";
      S.cancelAdn.hidden = true;
    }
  }

  async function sSubmit(e) {
    e.preventDefault();
    const file = S.imageInput.files && S.imageInput.files[0];
    if (!file) { sSetStatus("Please select an image first.","error"); return; }
    S.submitBtn.disabled = true;
    sShowPanel("loader");
    sStartLoader();
    sSetStatus("Running DINOv2 pipeline…","loading");
    const fd = new FormData();
    fd.append("image", file);
    fd.append("top_k","1");
    try {
      const r = await fetch(SEARCH_ENDPOINT, { method:"POST", body: fd });
      const p = await r.json();
      if (!r.ok || !p.success) {
        if (p.error === "no_confident_match") {
          const hint = p.best_similarity_pct != null
            ? ` Best: ${Number(p.best_similarity_pct).toFixed(1)}% (threshold ${(p.threshold*100).toFixed(0)}%)`
            : "";
          sResetResult();
          sSetStatus("No confident match found." + hint,"error");
          sShowPanel("empty");
          return;
        }
        throw new Error(p.error || p.message || "Search failed");
      }
      const m = (p.results && p.results[0]) || (p.summary && p.summary.best_match);
      if (!m) throw new Error("No match returned.");
      sRenderMatch(m);
      sSetStatus(`Match found — similarity ${formatPct(simPctFromMatch(m))}.`,"success");
    } catch(err) {
      sResetResult();
      sSetStatus(err.message || String(err),"error");
    } finally {
      sStopLoader();
      S.submitBtn.disabled = false;
    }
  }

  async function sHandleInlineAdnClick(e) {
    e.preventDefault();
    const url = S.adnBtn.getAttribute("href");
    if (!url || url === "#") return;
    if (!S.adnPanel.hidden && sLoadedAdnUrl === url) {
      S.adnPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
      sSetStatus(`ADN ready for ${S.resultRef.textContent || "this match"}.`, "success");
      return;
    }

    sAdnCtrl = new AbortController();
    S.adnBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 14l4-4 3 3 5-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Loading ADN...`;
    S.adnBtn.style.pointerEvents = "none";
    S.cancelAdn.hidden = false;
    sSetStatus("Loading ADN for the closest match...","loading");

    try {
      const r = await fetch(url, { headers:{Accept:"application/json"}, signal: sAdnCtrl.signal });
      const p = await r.json();
      if (!r.ok || !p.success || !p.adn_markdown) throw new Error(p.error || p.message || "ADN retrieval failed");
      S.adnContent.innerHTML = markdownToHtml(p.adn_markdown);
      S.adnMeta.textContent = `${labelForDomain(p.domain)} - ${p.reference || S.resultRef.textContent || "Reference unavailable"}`;
      S.adnPanel.hidden = false;
      S.dlPanel.hidden = true;
      sLoadedAdnUrl = url;
      S.adnPanel.scrollIntoView({ behavior: "smooth", block: "start" });
      sSetStatus(`ADN loaded for ${p.reference || S.resultRef.textContent || "the closest match"}.`, "success");
    } catch(err) {
      if (err.name !== "AbortError") S.adnPanel.hidden = true;
      sSetStatus(err.name === "AbortError" ? "Cancelled." : (err.message || String(err)),
                 err.name === "AbortError" ? "warning" : "error");
    } finally {
      sAdnCtrl = null;
      S.adnBtn.innerHTML = `<svg viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="M4 14l4-4 3 3 5-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg> Generate / See ADN`;
      S.adnBtn.style.pointerEvents = "";
      S.cancelAdn.hidden = true;
    }
  }

  function sAttachDropzone() {
    S.dropzone.addEventListener("click", () => S.imageInput.click());
    ["dragenter","dragover"].forEach(ev => S.dropzone.addEventListener(ev, e => { e.preventDefault(); S.dropzone.classList.add("is-active"); }));
    ["dragleave","dragend"].forEach(ev  => S.dropzone.addEventListener(ev, e => { e.preventDefault(); S.dropzone.classList.remove("is-active"); }));
    S.dropzone.addEventListener("drop", e => {
      e.preventDefault(); S.dropzone.classList.remove("is-active");
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || !files.length) return;
      try { const dt = new DataTransfer(); dt.items.add(files[0]); S.imageInput.files = dt.files; } catch(_) {}
      sRenderPreview(files[0]); sResetResult(); sSetStatus("Image loaded. Click Find Closest Match.",null);
    });
  }

  S.imageInput.addEventListener("change", () => { const f = S.imageInput.files[0]; if(f) { sRenderPreview(f); sResetResult(); sSetStatus("Image ready. Click Find Closest Match.",null); } });
  S.clearBtn.addEventListener("click", sClearPreview);
  S.form.addEventListener("submit", sSubmit);
  S.adnBtn.addEventListener("click", sHandleInlineAdnClick);
  S.cancelAdn.addEventListener("click", () => { if(sAdnCtrl) sAdnCtrl.abort(); });
  sAttachDropzone();
  sResetResult();
  sSetStatus("Waiting for an image.",null);


  // ══════════════════════════════════════════════════════════════════════════
  //  VIEW 2 — COMPARE ADN
  // ══════════════════════════════════════════════════════════════════════════

  const C = {
    inputA:        document.getElementById("cmpInputA"),
    inputB:        document.getElementById("cmpInputB"),
    dropzoneA:     document.getElementById("cmpDropzoneA"),
    dropzoneB:     document.getElementById("cmpDropzoneB"),
    previewWrapA:  document.getElementById("cmpPreviewWrapA"),
    previewWrapB:  document.getElementById("cmpPreviewWrapB"),
    previewA:      document.getElementById("cmpPreviewA"),
    previewB:      document.getElementById("cmpPreviewB"),
    clearA:        document.getElementById("cmpClearA"),
    clearB:        document.getElementById("cmpClearB"),
    nameA:         document.getElementById("cmpNameA"),
    nameB:         document.getElementById("cmpNameB"),
    sizeA:         document.getElementById("cmpSizeA"),
    sizeB:         document.getElementById("cmpSizeB"),
    matchA:        document.getElementById("cmpMatchA"),
    matchB:        document.getElementById("cmpMatchB"),
    matchDomainA:  document.getElementById("cmpMatchDomainA"),
    matchDomainB:  document.getElementById("cmpMatchDomainB"),
    matchRefA:     document.getElementById("cmpMatchRefA"),
    matchRefB:     document.getElementById("cmpMatchRefB"),
    matchNameA:    document.getElementById("cmpMatchNameA"),
    matchNameB:    document.getElementById("cmpMatchNameB"),
    matchScoreA:   document.getElementById("cmpMatchScoreA"),
    matchScoreB:   document.getElementById("cmpMatchScoreB"),
    compareBtn:    document.getElementById("compareBtn"),
    statusBar:     document.getElementById("cmpStatusBar"),
    statusText:    document.getElementById("cmpStatusText"),
    resultArea:    document.getElementById("cmpResultArea"),
    bannerRefA:    document.getElementById("cmpBannerRefA"),
    bannerRefB:    document.getElementById("cmpBannerRefB"),
    bannerScoreA:  document.getElementById("cmpBannerScoreA"),
    bannerScoreB:  document.getElementById("cmpBannerScoreB"),
    cosineSim:     document.getElementById("cmpCosineSim"),
    cosineBarFill: document.getElementById("cosineBarFill"),
    tabs:          document.querySelectorAll(".cmp-tab"),
    panels:        { diff: document.getElementById("cmpPanel-diff"), a: document.getElementById("cmpPanel-a"), b: document.getElementById("cmpPanel-b") },
    loaderDiff:    document.getElementById("cmpLoaderDiff"),
    loaderText:    document.getElementById("cmpLoaderText"),
    diffContent:   document.getElementById("cmpDiffContent"),
    adnA:          document.getElementById("cmpAdnA"),
    adnB:          document.getElementById("cmpAdnB"),
    dlRow:         document.getElementById("cmpDownloadRow"),
    dlLink:        document.getElementById("cmpDownloadLink"),
  };

  let cPreviewUrlA = null;
  let cPreviewUrlB = null;
  let cLoopId      = null;

  function cSetStatus(msg, tone) {
    C.statusBar.hidden = false;
    C.statusText.textContent = msg;
    C.statusBar.classList.remove("is-success","is-error","is-warning","is-loading");
    if (tone) C.statusBar.classList.add("is-" + tone);
  }

  function cUpdateCompareBtn() {
    const aReady = !!(C.inputA.files && C.inputA.files[0]);
    const bReady = !!(C.inputB.files && C.inputB.files[0]);
    C.compareBtn.disabled = !(aReady && bReady);
  }

  function cRenderPreview(side, file) {
    const isA = side === "A";
    const urlRef    = isA ? "cPreviewUrlA" : "cPreviewUrlB";
    const prevImg   = isA ? C.previewA    : C.previewB;
    const prevWrap  = isA ? C.previewWrapA: C.previewWrapB;
    const dzBtn     = isA ? C.dropzoneA   : C.dropzoneB;
    const nameEl    = isA ? C.nameA       : C.nameB;
    const sizeEl    = isA ? C.sizeA       : C.sizeB;

    if (isA && cPreviewUrlA) { URL.revokeObjectURL(cPreviewUrlA); cPreviewUrlA = null; }
    if (!isA && cPreviewUrlB) { URL.revokeObjectURL(cPreviewUrlB); cPreviewUrlB = null; }

    if (!file) { prevWrap.hidden = true; dzBtn.hidden = false; return; }
    const objUrl = URL.createObjectURL(file);
    if (isA) cPreviewUrlA = objUrl; else cPreviewUrlB = objUrl;
    prevImg.src = objUrl;
    nameEl.textContent = file.name;
    sizeEl.textContent = formatBytes(file.size);
    prevWrap.hidden = false;
    dzBtn.hidden    = true;
  }

  function cClear(side) {
    const input = side === "A" ? C.inputA : C.inputB;
    const match = side === "A" ? C.matchA : C.matchB;
    input.value = "";
    cRenderPreview(side, null);
    match.hidden = true;
    cUpdateCompareBtn();
    C.resultArea.hidden = true;
    C.statusBar.hidden  = true;
    C.dlRow.hidden      = true;
  }

  function cAttachDropzone(side) {
    const dz    = side === "A" ? C.dropzoneA : C.dropzoneB;
    const input = side === "A" ? C.inputA    : C.inputB;
    dz.addEventListener("click", () => input.click());
    ["dragenter","dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("is-active"); }));
    ["dragleave","dragend"].forEach(ev  => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("is-active"); }));
    dz.addEventListener("drop", e => {
      e.preventDefault(); dz.classList.remove("is-active");
      const files = e.dataTransfer && e.dataTransfer.files;
      if (!files || !files.length) return;
      try { const dt = new DataTransfer(); dt.items.add(files[0]); input.files = dt.files; } catch(_) {}
      cRenderPreview(side, files[0]);
      cUpdateCompareBtn();
    });
    input.addEventListener("change", () => {
      const f = input.files && input.files[0];
      if (f) { cRenderPreview(side, f); cUpdateCompareBtn(); }
    });
  }

  function cShowMatchBadge(side, match) {
    const domEl   = side === "A" ? C.matchDomainA : C.matchDomainB;
    const refEl   = side === "A" ? C.matchRefA    : C.matchRefB;
    const nameEl  = side === "A" ? C.matchNameA   : C.matchNameB;
    const scoreEl = side === "A" ? C.matchScoreA  : C.matchScoreB;
    const badge   = side === "A" ? C.matchA       : C.matchB;
    const domain = (match.domain || "").toLowerCase();
    domEl.textContent  = labelForDomain(domain);
    refEl.textContent  = match.reference || "—";
    nameEl.textContent = match.display_name || match.material_name || "—";
    scoreEl.textContent = formatPct(simPctFromMatch(match));
    badge.hidden = false;
  }

  function cSwitchTab(tabId) {
    C.tabs.forEach(t => {
      const active = t.dataset.tab === tabId;
      t.classList.toggle("is-active", active);
      t.setAttribute("aria-selected", String(active));
    });
    Object.entries(C.panels).forEach(([id, panel]) => { panel.hidden = id !== tabId; });
  }

  C.tabs.forEach(t => t.addEventListener("click", () => cSwitchTab(t.dataset.tab)));
  C.clearA.addEventListener("click", () => cClear("A"));
  C.clearB.addEventListener("click", () => cClear("B"));
  cAttachDropzone("A");
  cAttachDropzone("B");

  function cStartLoader(msg) {
    let i = 0;
    if (C.loaderText) C.loaderText.textContent = COMPARE_STEPS[0];
    cLoopId = setInterval(() => {
      i = (i + 1) % COMPARE_STEPS.length;
      if (C.loaderText) C.loaderText.textContent = COMPARE_STEPS[i];
    }, 1200);
  }
  function cStopLoader() { if (cLoopId) { clearInterval(cLoopId); cLoopId = null; } }

  async function cRunCompare() {
    const fileA = C.inputA.files && C.inputA.files[0];
    const fileB = C.inputB.files && C.inputB.files[0];
    if (!fileA || !fileB) { cSetStatus("Please upload both images.","error"); return; }

    C.compareBtn.disabled = true;
    C.resultArea.hidden   = true;
    C.dlRow.hidden        = true;
    C.matchA.hidden       = true;
    C.matchB.hidden       = true;
    cSetStatus("Running DINOv2 pipeline on both images…","loading");

    // Show diff loader
    C.loaderDiff.hidden  = false;
    C.diffContent.innerHTML = "";
    C.adnA.innerHTML = "";
    C.adnB.innerHTML = "";
    cStartLoader();

    const fd = new FormData();
    fd.append("image_a", fileA);
    fd.append("image_b", fileB);

    try {
      const r = await fetch(COMPARE_ENDPOINT, { method:"POST", body: fd });
      const p = await r.json();

      if (!r.ok || !p.success) {
        throw new Error(p.error || p.message || "Compare failed");
      }

      const mA = p.match_a;
      const mB = p.match_b;

      // Show match badges in the upload cards
      cShowMatchBadge("A", mA);
      cShowMatchBadge("B", mB);

      // Score banner
      C.bannerRefA.textContent   = mA.reference || "—";
      C.bannerRefB.textContent   = mB.reference || "—";
      C.bannerScoreA.textContent = `${formatPct(simPctFromMatch(mA))} similarity`;
      C.bannerScoreB.textContent = `${formatPct(simPctFromMatch(mB))} similarity`;

      // Cross-embedding similarity
      const cosPct = p.cross_similarity_pct != null ? p.cross_similarity_pct : (p.cross_similarity != null ? p.cross_similarity * 100 : null);
      C.cosineSim.textContent = formatPct(cosPct);
      C.cosineBarFill.style.width = "0%";
      requestAnimationFrame(() => requestAnimationFrame(() => {
        C.cosineBarFill.style.width = (cosPct != null ? Math.min(100, Math.max(0, cosPct)) : 0) + "%";
      }));

      // ADN content
      C.diffContent.innerHTML = markdownToHtml(p.adn_diff  || p.diff_report || "No differential content returned.");
      C.adnA.innerHTML        = markdownToHtml(p.adn_a     || "ADN for image A not available.");
      C.adnB.innerHTML        = markdownToHtml(p.adn_b     || "ADN for image B not available.");

      // Show result area
      C.resultArea.hidden = false;
      cSwitchTab("diff");

      // Download
      if (p.download_url || p.absolute_url) {
        C.dlLink.href = p.download_url || p.absolute_url;
        C.dlRow.hidden = false;
      }

      cSetStatus(
        `Comparison complete — cross-embedding similarity: ${formatPct(cosPct)}`,
        "success"
      );

    } catch(err) {
      cSetStatus(err.message || String(err),"error");
    } finally {
      cStopLoader();
      C.loaderDiff.hidden   = true;
      C.compareBtn.disabled = false;
    }
  }

  C.compareBtn.addEventListener("click", cRunCompare);

})();
