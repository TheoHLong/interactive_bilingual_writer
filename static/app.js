(function () {
  const SETTINGS_KEY = "interactive-writer-settings";
  const DRAFT_KEY = "interactive-writer-source";

  const els = {
    direction: document.getElementById("direction"),
    debounceSeconds: document.getElementById("debounceSeconds"),
    minChars: document.getElementById("minChars"),
    includeActive: document.getElementById("includeActive"),
    exportFormat: document.getElementById("exportFormat"),
    sourceInput: document.getElementById("sourceInput"),
    sourceCount: document.getElementById("sourceCount"),
    lastRun: document.getElementById("lastRun"),
    status: document.getElementById("status"),
    preview: document.getElementById("preview"),
    targetLabel: document.getElementById("targetLabel"),
    stats: document.getElementById("stats"),
    modelLine: document.getElementById("modelLine"),
    translateNow: document.getElementById("translateNow"),
    refineNow: document.getElementById("refineNow"),
    saveSnapshot: document.getElementById("saveSnapshot")
  };

  let debounceTimer = null;
  let requestId = 0;
  let currentAbort = null;
  let lastSourceSent = "";
  let latestTranslation = "";
  let latestLangs = { source: "Source", target: "Translation" };
  let latestSourceBlocks = [];
  let latestPreviewBlocks = [];
  let selectedPreviewBlockId = null;
  let isComposing = false;

  function loadInitialState() {
    try {
      const settings = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
      if (settings.direction) els.direction.value = settings.direction;
      if (settings.debounceSeconds) els.debounceSeconds.value = settings.debounceSeconds;
      if (settings.minChars !== undefined) els.minChars.value = settings.minChars;
      if (settings.includeActive !== undefined) els.includeActive.checked = Boolean(settings.includeActive);
      if (settings.exportFormat) els.exportFormat.value = settings.exportFormat;
    } catch (_error) {
      // Ignore malformed local settings.
    }

    const draft = localStorage.getItem(DRAFT_KEY);
    if (draft) {
      els.sourceInput.value = draft;
    }
    updateCounts();
  }

  function saveSettings() {
    const settings = {
      direction: els.direction.value,
      debounceSeconds: Number(els.debounceSeconds.value),
      minChars: Number(els.minChars.value),
      includeActive: els.includeActive.checked,
      exportFormat: els.exportFormat.value
    };
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  }

  function setStatus(message, tone) {
    els.status.textContent = message;
    els.status.classList.toggle("is-busy", tone === "busy");
    els.status.classList.toggle("is-error", tone === "error");
  }

  function updateCounts() {
    const count = els.sourceInput.value.length;
    els.sourceCount.textContent = `${count} char${count === 1 ? "" : "s"}`;
  }

  function detectLanguage(text) {
    const compact = text.replace(/\s+/g, "");
    if (!compact) return "English";
    const cjk = compact.match(/[\u4e00-\u9fff]/g) || [];
    const latin = compact.match(/[A-Za-z]/g) || [];
    const total = cjk.length + latin.length;
    if (total === 0) return "English";
    return cjk.length / total >= 0.2 ? "Chinese" : "English";
  }

  function effectiveMinChars(text) {
    const configured = Math.max(0, Number(els.minChars.value) || 0);
    if (detectLanguage(text) === "Chinese") {
      return Math.max(1, Math.round(configured / 2));
    }
    return configured;
  }

  function scheduleTranslation() {
    if (isComposing) return;
    clearTimeout(debounceTimer);
    const delayMs = Math.max(0.5, Number(els.debounceSeconds.value) || 2) * 1000;
    debounceTimer = setTimeout(() => runTranslation({ force: false, mode: "quick" }), delayMs);
  }

  function changedChars(a, b) {
    if (!b) return a.trim().length;
    let start = 0;
    while (start < a.length && start < b.length && a[start] === b[start]) {
      start += 1;
    }
    let endA = a.length - 1;
    let endB = b.length - 1;
    while (endA >= start && endB >= start && a[endA] === b[endB]) {
      endA -= 1;
      endB -= 1;
    }
    return Math.max(endA - start + 1, endB - start + 1, 0);
  }

  function splitBlocks(text, cursor) {
    if (!text.trim()) return [];

    const blocks = [];
    const lines = text.split("\n");
    let current = [];
    let currentStart = 0;
    let position = 0;
    let inFence = false;

    function pushBlock(endPosition) {
      const blockText = current.join("\n").trimEnd();
      if (blockText.trim()) {
        blocks.push({
          id: String(blocks.length),
          text: blockText,
          start: currentStart,
          end: endPosition,
          active: cursor >= currentStart && cursor <= endPosition
        });
      }
      current = [];
    }

    lines.forEach((line) => {
      const lineStart = position;
      const lineEnd = position + line.length;
      const trimmed = line.trim();

      if (current.length === 0) {
        currentStart = lineStart;
      }

      const wasInFence = inFence;
      if (trimmed.startsWith("```")) {
        inFence = !inFence;
      }
      if (wasInFence && !inFence) {
        current.push(line);
        pushBlock(lineEnd);
        position = lineEnd + 1;
        return;
      }

      if (!inFence && trimmed === "") {
        pushBlock(lineStart);
      } else {
        current.push(line);
      }

      position = lineEnd + 1;
    });

    if (current.length > 0) {
      pushBlock(text.length);
    }

    return blocks;
  }

  async function runTranslation(options) {
    const force = Boolean(options.force);
    const mode = options.mode || "quick";
    if (isComposing && !force) return;
    const sourceText = els.sourceInput.value;
    const blocks = splitBlocks(sourceText, els.sourceInput.selectionStart || 0);
    latestSourceBlocks = blocks;
    const activeBlock = blocks.find((block) => block.active);
    const thresholdText = activeBlock ? activeBlock.text : sourceText;
    const minChars = effectiveMinChars(thresholdText);
    const delta = changedChars(sourceText, lastSourceSent);

    if (!force && lastSourceSent && delta <= 0) {
      return;
    }
    if (!force && lastSourceSent && delta < minChars) {
      setStatus(`Waiting for ${minChars - delta} more changed chars`, null);
      return;
    }

    const includeActive = els.includeActive.checked;
    const segments = blocks
      .filter((block) => includeActive || !block.active)
      .map((block) => ({ id: block.id, text: block.text }));

    if (segments.length === 0) {
      latestTranslation = "";
      latestPreviewBlocks = [];
      renderPreview("");
      setStatus("Idle", null);
      return;
    }

    if (currentAbort) {
      currentAbort.abort();
    }
    currentAbort = new AbortController();
    const myRequestId = (requestId += 1);

    setStatus(mode === "refine" ? "Refining" : "Translating", "busy");
    setButtonsDisabled(true);

    try {
      const response = await fetch("/api/translate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          direction: els.direction.value,
          mode,
          source_text: sourceText,
          export_format: els.exportFormat.value,
          segments
        }),
        signal: currentAbort.signal
      });

      if (!response.ok) {
        throw new Error("Translation failed.");
      }
      if (!response.body) {
        throw new Error("Streaming response is not available.");
      }

      const includedIds = new Set(segments.map((segment) => segment.id));
      const partialById = new Map();
      const doneById = new Map();
      const decoder = new TextDecoder();
      const reader = response.body.getReader();
      let buffer = "";
      let finalData = null;
      let streamError = null;

      const renderStreamState = () => {
        const previewBlocks = blocks.map((block) => {
          let markdown = "*Editing...*";
          if (doneById.has(block.id)) {
            markdown = doneById.get(block.id);
          } else if (partialById.has(block.id)) {
            markdown = partialById.get(block.id) || "...";
          } else if (includedIds.has(block.id)) {
            markdown = "...";
          }
          return {
            id: block.id,
            markdown,
            sourceStart: block.start,
            sourceEnd: block.end,
            sourceText: block.text
          };
        });
        latestTranslation = previewBlocks.map((block) => block.markdown).join("\n\n");
        latestPreviewBlocks = previewBlocks;
        renderPreviewBlocks(previewBlocks);
      };

      const handleEvent = (event) => {
        if (event.type === "meta") {
          updateResponseMeta({
            source_lang: event.source_lang,
            target_lang: event.target_lang,
            model: event.model,
            stats: {}
          });
          return;
        }
        if (event.type === "segment_status") {
          const statusLabels = {
            quick: "Preparing quick draft",
            reflecting: "Reflecting",
            refine: "Refining"
          };
          setStatus(statusLabels[event.status] || "Translating", "busy");
          return;
        }
        if (event.type === "segment_start") {
          partialById.set(event.id, "");
          renderStreamState();
          return;
        }
        if (event.type === "chunk") {
          partialById.set(event.id, (partialById.get(event.id) || "") + event.text);
          renderStreamState();
          return;
        }
        if (event.type === "segment_done") {
          partialById.delete(event.id);
          doneById.set(event.id, event.translation || "");
          renderStreamState();
          return;
        }
        if (event.type === "done") {
          finalData = event;
          return;
        }
        if (event.type === "error") {
          streamError = event.message || "Translation failed.";
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          handleEvent(JSON.parse(line));
          if (streamError) throw new Error(streamError);
        }
        if (done) break;
      }
      if (buffer.trim()) {
        handleEvent(JSON.parse(buffer));
      }
      if (streamError) {
        throw new Error(streamError);
      }
      if (!finalData) {
        throw new Error("Translation stream ended before completion.");
      }
      if (myRequestId !== requestId) return;

      latestLangs = { source: finalData.source_lang, target: finalData.target_lang };
      lastSourceSent = sourceText;
      updateResponseMeta(finalData);
      localStorage.setItem(DRAFT_KEY, sourceText);
      els.lastRun.textContent = new Date().toLocaleTimeString();
      setStatus("Translated", null);
    } catch (error) {
      if (error.name !== "AbortError") {
        setStatus(error.message, "error");
        checkHealth();
      }
    } finally {
      if (myRequestId === requestId) {
        setButtonsDisabled(false);
      }
    }
  }

  function updateResponseMeta(data) {
    els.targetLabel.textContent = `${data.source_lang} to ${data.target_lang}`;
    els.modelLine.textContent = `Ollama model: ${data.model}`;
    const stats = data.stats || {};
    const quickReused = stats.quick_reused ? `, ${stats.quick_reused} quick reused` : "";
    const warnings = stats.warnings ? `, ${stats.warnings} structure warnings` : "";
    els.stats.textContent = `${stats.translated || 0} translated, ${stats.cached || 0} cached${quickReused}${warnings}`;
    if (data.structure_warnings && data.structure_warnings.length > 0) {
      els.stats.title = data.structure_warnings
        .map((item) => `Segment ${item.id}: ${item.warnings.join(", ")}`)
        .join("\n");
    } else {
      els.stats.removeAttribute("title");
    }
  }

  function setButtonsDisabled(disabled) {
    els.translateNow.disabled = disabled;
    els.refineNow.disabled = disabled;
    els.saveSnapshot.disabled = disabled;
  }

  function renderPreview(markdown) {
    if (!markdown.trim()) {
      clearPreview();
      return;
    }
    els.preview.classList.remove("empty");
    els.preview.innerHTML = renderMarkdown(markdown);
    highlightPreview();
  }

  function renderPreviewBlocks(blocks) {
    if (!blocks.some((block) => block.markdown.trim())) {
      clearPreview();
      return;
    }

    els.preview.classList.remove("empty");
    els.preview.innerHTML = blocks
      .map((block) => {
        const isSelected = String(block.id) === selectedPreviewBlockId;
        return [
          `<section class="preview-block${isSelected ? " is-selected" : ""}"`,
          ` data-block-id="${escapeAttribute(block.id)}"`,
          ` data-source-start="${escapeAttribute(block.sourceStart)}"`,
          ` data-source-end="${escapeAttribute(block.sourceEnd)}" tabindex="0">`,
          renderMarkdown(block.markdown),
          "</section>"
        ].join("");
      })
      .join("");
    highlightPreview();
  }

  function clearPreview() {
    els.preview.classList.add("empty");
    els.preview.innerHTML = "";
    latestPreviewBlocks = [];
  }

  function renderMarkdown(markdown) {
    if (window.marked && typeof window.marked.parse === "function") {
      return window.marked.parse(markdown, {
        async: false,
        breaks: false,
        gfm: true
      });
    }
    return `<pre>${escapeHtml(markdown)}</pre>`;
  }

  function jumpToSourceBlock(blockId) {
    const block = findSourceBlockForPreview(blockId);
    if (!block) return;

    selectedPreviewBlockId = String(blockId);
    updatePreviewSelection();

    const cursorPosition = Math.min(block.start, els.sourceInput.value.length);
    els.sourceInput.setSelectionRange(cursorPosition, cursorPosition);
    scrollSourceToPosition(cursorPosition);
    try {
      els.sourceInput.focus({ preventScroll: true });
    } catch (_error) {
      els.sourceInput.focus();
    }
  }

  function findSourceBlockForPreview(blockId) {
    const id = String(blockId);
    const currentBlocks = splitBlocks(els.sourceInput.value, -1);
    const previewBlock = latestPreviewBlocks.find((item) => item.id === id);

    if (previewBlock) {
      const exactMatches = currentBlocks.filter((block) => block.text === previewBlock.sourceText);
      if (exactMatches.length > 0) {
        return exactMatches.reduce((best, block) => {
          const bestDistance = Math.abs(best.start - previewBlock.sourceStart);
          const blockDistance = Math.abs(block.start - previewBlock.sourceStart);
          return blockDistance < bestDistance ? block : best;
        });
      }

      const currentById = currentBlocks.find((block) => block.id === id);
      if (currentById) return currentById;

      const currentByOffset = currentBlocks.find((block) => (
        previewBlock.sourceStart >= block.start && previewBlock.sourceStart <= block.end
      ));
      if (currentByOffset) return currentByOffset;

      return {
        id,
        text: previewBlock.sourceText,
        start: Math.min(previewBlock.sourceStart, els.sourceInput.value.length),
        end: Math.min(previewBlock.sourceEnd, els.sourceInput.value.length),
        active: false
      };
    }

    return currentBlocks.find((block) => block.id === id) ||
      latestSourceBlocks.find((block) => block.id === id);
  }

  function scrollSourceToPosition(position) {
    const caretTop = measureTextareaCaretTop(position);
    const targetTop = Math.max(0, caretTop - els.sourceInput.clientHeight * 0.22);
    els.sourceInput.scrollTo({ top: targetTop, behavior: "smooth" });
  }

  function measureTextareaCaretTop(position) {
    const styles = window.getComputedStyle(els.sourceInput);
    const mirror = document.createElement("div");
    const copiedProperties = [
      "boxSizing",
      "fontFamily",
      "fontSize",
      "fontStyle",
      "fontWeight",
      "letterSpacing",
      "lineHeight",
      "paddingTop",
      "paddingRight",
      "paddingBottom",
      "paddingLeft",
      "tabSize",
      "textAlign",
      "textIndent",
      "textTransform",
      "wordSpacing"
    ];

    copiedProperties.forEach((property) => {
      mirror.style[property] = styles[property];
    });
    mirror.style.position = "absolute";
    mirror.style.visibility = "hidden";
    mirror.style.left = "-9999px";
    mirror.style.top = "0";
    mirror.style.width = `${els.sourceInput.clientWidth}px`;
    mirror.style.whiteSpace = "pre-wrap";
    mirror.style.overflowWrap = "break-word";
    mirror.style.wordBreak = "normal";
    mirror.style.border = "0";

    const marker = document.createElement("span");
    marker.textContent = "\u200b";
    mirror.textContent = els.sourceInput.value.slice(0, position);
    mirror.appendChild(marker);
    document.body.appendChild(mirror);

    const caretTop = marker.offsetTop;
    mirror.remove();
    return caretTop;
  }

  function updatePreviewSelection() {
    els.preview.querySelectorAll(".preview-block").forEach((block) => {
      block.classList.toggle("is-selected", block.dataset.blockId === selectedPreviewBlockId);
    });
  }

  function escapeAttribute(value) {
    return escapeHtml(String(value));
  }

  function highlightPreview() {
    if (!window.hljs || typeof window.hljs.highlightElement !== "function") return;
    els.preview.querySelectorAll("pre code").forEach((block) => {
      window.hljs.highlightElement(block);
    });
  }

  function escapeHtml(value) {
    return value
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  async function saveSnapshot() {
    try {
      const response = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: els.sourceInput.value,
          translation: latestTranslation,
          source_lang: latestLangs.source,
          target_lang: latestLangs.target,
          export_format: els.exportFormat.value
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Save failed.");
      }
      setStatus(`Saved ${data.path}`, null);
    } catch (error) {
      setStatus(error.message, "error");
    }
  }

  els.sourceInput.addEventListener("input", () => {
    updateCounts();
    localStorage.setItem(DRAFT_KEY, els.sourceInput.value);
    scheduleTranslation();
  });

  els.sourceInput.addEventListener("compositionstart", () => {
    isComposing = true;
    clearTimeout(debounceTimer);
  });

  els.sourceInput.addEventListener("compositionend", () => {
    isComposing = false;
    updateCounts();
    localStorage.setItem(DRAFT_KEY, els.sourceInput.value);
    scheduleTranslation();
  });

  [els.direction, els.debounceSeconds, els.minChars, els.includeActive, els.exportFormat].forEach((control) => {
    control.addEventListener("change", () => {
      saveSettings();
      if (control !== els.exportFormat) {
        scheduleTranslation();
      }
    });
  });

  async function checkHealth() {
    try {
      const response = await fetch("/api/health");
      const data = await response.json();
      els.modelLine.textContent = `Ollama model: ${data.model}`;
      if (!data.ok) {
        setStatus(data.error || "Ollama unavailable", "error");
      }
    } catch (_error) {
      setStatus("Health check failed", "error");
    }
  }

  els.translateNow.addEventListener("click", () => runTranslation({ force: true, mode: "quick" }));
  els.refineNow.addEventListener("click", () => runTranslation({ force: true, mode: "refine" }));
  els.saveSnapshot.addEventListener("click", saveSnapshot);

  els.preview.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const previewBlock = event.target.closest(".preview-block");
    if (!previewBlock || !els.preview.contains(previewBlock)) return;
    event.preventDefault();
    jumpToSourceBlock(previewBlock.dataset.blockId);
  });

  els.preview.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    if (!(event.target instanceof Element)) return;
    const previewBlock = event.target.closest(".preview-block");
    if (!previewBlock || !els.preview.contains(previewBlock)) return;
    event.preventDefault();
    jumpToSourceBlock(previewBlock.dataset.blockId);
  });

  loadInitialState();
  checkHealth();
  renderPreview("");
  if (els.sourceInput.value.trim()) {
    scheduleTranslation();
  }
})();
