const MAX_REFERENCE_BYTES = 20 * 1024 * 1024;
const MAX_SEED = 9223372036854775807n;
const REQUEST_TIMEOUT_MS = 5 * 60 * 1000;
const SETTINGS_SAVE_DELAY_MS = 180;
const WIDE_LAYOUT_QUERY = "(min-width: 1180px)";
const STICKY_COLUMN_TOP_PX = 20;
const storage = globalThis.PuLIDStorage;
const wideLayout = window.matchMedia(WIDE_LAYOUT_QUERY);

const state = {
  engine: "sdxl",
  engineSettings: {},
  ready: false,
  inventoryRequest: 0,
  inventory: null,
  resultUrl: null,
  referenceUrl: null,
  generating: false,
  settingsSaveTimer: null,
};

const elements = {
  engine: document.querySelector("#engine"),
  engineHint: document.querySelector("#engineHint"),
  denoise: document.querySelector("#denoise"),
  backendUrl: document.querySelector("#backendUrl"),
  reconnectButton: document.querySelector("#reconnectButton"),
  connectionState: document.querySelector("#connectionState"),
  connectionLabel: document.querySelector("#connectionLabel"),
  form: document.querySelector("#generationForm"),
  reference: document.querySelector("#reference"),
  referencePreview: document.querySelector("#referencePreview"),
  uploadZone: document.querySelector("#uploadZone"),
  uploadTitle: document.querySelector("#uploadTitle"),
  uploadHint: document.querySelector("#uploadHint"),
  clearReference: document.querySelector("#clearReference"),
  character: document.querySelector("#character"),
  prompt: document.querySelector("#prompt"),
  promptCount: document.querySelector("#promptCount"),
  promptLimit: document.querySelector("#promptLimit"),
  negativeMode: document.querySelector("#negativeMode"),
  negativePrompt: document.querySelector("#negativePrompt"),
  negativePromptField: document.querySelector("#negativePromptField"),
  negativeCount: document.querySelector("#negativeCount"),
  model: document.querySelector("#model"),
  modelLabel: document.querySelector("#modelLabel"),
  textEncoder: document.querySelector("#textEncoder"),
  method: document.querySelector("#method"),
  sigmas: document.querySelector("#sigmas"),
  clipSkip2: document.querySelector("#clipSkip2"),
  cfg: document.querySelector("#cfg"),
  steps: document.querySelector("#steps"),
  strength: document.querySelector("#strength"),
  seed: document.querySelector("#seed"),
  generateButton: document.querySelector("#generateButton"),
  formError: document.querySelector("#formError"),
  emptyResult: document.querySelector("#emptyResult"),
  generatingResult: document.querySelector("#generatingResult"),
  generatedImage: document.querySelector("#generatedImage"),
  imageLightbox: document.querySelector("#imageLightbox"),
  lightboxImage: document.querySelector("#lightboxImage"),
  closeLightbox: document.querySelector("#closeLightbox"),
  resultMeta: document.querySelector("#resultMeta"),
  resultSeed: document.querySelector("#resultSeed"),
  resultModel: document.querySelector("#resultModel"),
  resultMethod: document.querySelector("#resultMethod"),
  resultSigmas: document.querySelector("#resultSigmas"),
  resolution: document.querySelector("#resolution"),
  downloadButton: document.querySelector("#downloadButton"),
  clearLocalData: document.querySelector("#clearLocalData"),
  advancedSettings: document.querySelector(".advanced"),
  cardColumns: [...document.querySelectorAll(".card-column")],
};

const columnScrollState = new WeakMap(
  elements.cardColumns.map((column) => [column, { offset: 0, maxOffset: 0 }]),
);
let columnLayoutFrame = null;

function resetColumnLayout(column, scrollState) {
  scrollState.offset = 0;
  scrollState.maxOffset = 0;
  column.scrollTop = 0;
  column.classList.remove("is-column-stuck", "is-column-pinned");
  for (const property of [
    "--column-height",
    "--column-left",
    "--column-width",
    "--column-offset",
  ]) {
    column.style.removeProperty(property);
  }
}

function updateColumnLayout() {
  columnLayoutFrame = null;
  for (const column of elements.cardColumns) {
    const scrollState = columnScrollState.get(column);
    const card = column.firstElementChild;
    if (!scrollState || !card || !wideLayout.matches) {
      if (scrollState) resetColumnLayout(column, scrollState);
      continue;
    }

    const columnBounds = column.getBoundingClientRect();
    const cardHeight = card.getBoundingClientRect().height;
    const visibleHeight = Math.max(0, window.innerHeight - STICKY_COLUMN_TOP_PX * 2);
    scrollState.maxOffset = Math.max(0, cardHeight - visibleHeight);
    scrollState.offset = Math.min(scrollState.offset, scrollState.maxOffset);

    const shouldPin = columnBounds.top <= STICKY_COLUMN_TOP_PX + 1;
    if (!shouldPin) scrollState.offset = 0;
    column.style.setProperty("--column-height", `${Math.ceil(cardHeight)}px`);
    column.style.setProperty("--column-left", `${columnBounds.left}px`);
    column.style.setProperty("--column-width", `${columnBounds.width}px`);
    column.style.setProperty("--column-offset", `${scrollState.offset}px`);
    column.classList.toggle("is-column-pinned", shouldPin);
  }
}

function queueColumnLayoutUpdate() {
  if (columnLayoutFrame !== null) return;
  columnLayoutFrame = window.requestAnimationFrame(updateColumnLayout);
}

function wheelDeltaInPixels(event) {
  if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) return event.deltaY * 16;
  if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) return event.deltaY * window.innerHeight;
  return event.deltaY;
}

function scrollPinnedColumn(event) {
  const column = event.currentTarget;
  const scrollState = columnScrollState.get(column);
  if (!wideLayout.matches || !column.classList.contains("is-column-pinned") || !scrollState) {
    return;
  }

  const delta = wheelDeltaInPixels(event);
  if (delta === 0) return;
  const nextOffset = Math.max(
    0,
    Math.min(scrollState.maxOffset, scrollState.offset + delta),
  );
  const keepsStickyWhileScrollingUp = delta < 0;
  if (nextOffset === scrollState.offset && !keepsStickyWhileScrollingUp) return;
  event.preventDefault();
  if (nextOffset === scrollState.offset) return;
  scrollState.offset = nextOffset;
  column.style.setProperty("--column-offset", `${nextOffset}px`);
}

const columnResizeObserver =
  "ResizeObserver" in window
    ? new ResizeObserver(queueColumnLayoutUpdate)
    : null;
for (const column of elements.cardColumns) {
  column.addEventListener("wheel", scrollPinnedColumn, { passive: false });
  if (column.firstElementChild) columnResizeObserver?.observe(column.firstElementChild);
}

function setConnection(status, label) {
  elements.connectionState.className = `connection-state ${status}`;
  elements.connectionLabel.textContent = label;
}

function setError(message = "") {
  elements.formError.textContent = message;
  elements.formError.hidden = !message;
}

function selectDefault(items) {
  return items.find((item) => item.default) ?? items[0] ?? null;
}

function fillSelect(select, items, selectedName = null) {
  select.replaceChildren();
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.name;
    option.textContent = item.label ?? item.filename ?? item.name;
    select.append(option);
  }
  const fallback = selectDefault(items)?.name ?? "";
  select.value = items.some((item) => item.name === selectedName) ? selectedName : fallback;
  select.disabled = items.length === 0;
}

function refreshSigmaOptions(selectedName = elements.sigmas.value) {
  if (state.engine === "krea2") return;
  if (!state.inventory) return;
  const method = state.inventory.sampling_methods.find(
    (item) => item.name === elements.method.value,
  );
  const supported = new Set(method?.supported_sigma_schedules ?? ["normal"]);
  const visible = state.inventory.sigma_schedules.filter((item) => supported.has(item.name));
  fillSelect(elements.sigmas, visible, selectedName);
}

function collectSettings() {
  return {
    engine: state.engine,
    engineSettings: { ...state.engineSettings, [state.engine]: engineValues() },
    character: elements.character.value,
    prompt: elements.prompt.value,
    negativeMode: elements.negativeMode.value,
    negativePrompt: elements.negativePrompt.value,
    clipSkip2: elements.clipSkip2.checked,
    model: elements.model.value,
    textEncoder: elements.textEncoder.value,
    cfg: elements.cfg.value,
    steps: elements.steps.value,
    strength: elements.strength.value,
    denoise: elements.denoise.value,
    method: elements.method.value,
    sigmas: elements.sigmas.value,
    seed: elements.seed.value,
    resolution: elements.resolution.value,
    advancedOpen: elements.advancedSettings.open,
  };
}

function applySettings(settings) {
  if (!settings || typeof settings !== "object") return;
  state.engine = settings.engine === "krea2" ? "krea2" : "sdxl";
  elements.engine.value = state.engine;
  state.engineSettings = settings.engineSettings ?? {};
  const textFields = {
    character: elements.character,
    prompt: elements.prompt,
    negativePrompt: elements.negativePrompt,
    cfg: elements.cfg,
    steps: elements.steps,
    strength: elements.strength,
    denoise: elements.denoise,
    seed: elements.seed,
  };
  for (const [name, element] of Object.entries(textFields)) {
    if (typeof settings[name] === "string") element.value = settings[name];
  }
  if (["default", "custom", "disabled"].includes(settings.negativeMode)) {
    elements.negativeMode.value = settings.negativeMode;
  }
  if (typeof settings.clipSkip2 === "boolean") {
    elements.clipSkip2.checked = settings.clipSkip2;
  }
  if (typeof settings.advancedOpen === "boolean") {
    elements.advancedSettings.open = settings.advancedOpen;
  }
  if (
    typeof settings.resolution === "string" &&
    [...elements.resolution.options].some((option) => option.value === settings.resolution)
  ) {
    elements.resolution.value = settings.resolution;
  }
  updateNegativePromptMode();
  updateCount(elements.prompt, elements.promptCount);
  updateCount(elements.negativePrompt, elements.negativeCount);
  validateSeed();
  updateEngineFields();
}

function engineValues() {
  return { cfg: elements.cfg.value, steps: elements.steps.value,
    resolution: elements.resolution.value, denoise: elements.denoise.value,
    model: elements.model.value, textEncoder: elements.textEncoder.value,
    method: elements.method.value, sigmas: elements.sigmas.value };
}

function updateEngineFields() {
  const krea = state.engine === "krea2";
  updatePromptValidation();
  for (const [selector, hidden] of [["[data-sdxl-only]", krea], ["[data-krea2-only]", !krea]]) {
    document.querySelectorAll(selector).forEach((group) => {
      group.hidden = hidden;
      if (group.tagName === "FIELDSET") group.disabled = hidden;
      group.querySelectorAll("input, select, textarea, button").forEach((input) => { input.disabled = hidden; });
    });
  }
  elements.modelLabel.textContent = krea ? "Modèle Krea v2" : "Modèle SDXL";
  elements.model.disabled = !state.ready || !state.inventory?.models?.length;
  elements.textEncoder.disabled = !krea || !state.ready || !state.inventory?.text_encoders?.length;
  elements.reference.setCustomValidity("");
  elements.engineHint.textContent = krea
    ? "Décrivez votre image. Aucune photo ni identité nécessaire."
    : "Utilise votre portrait de référence.";
  document.querySelector("#engineEyebrow").textContent = krea ? "Krea v2" : "SDXL + PuLID";
  document.querySelector("#pageTitle").textContent = krea
    ? "De vos mots à l’image."
    : "Créez une image qui garde son identité.";
  updateNegativePromptMode();
  queueColumnLayoutUpdate();
}

function changeEngine() {
  state.engineSettings[state.engine] = engineValues();
  state.engine = elements.engine.value;
  const defaults = state.engine === "krea2"
    ? { cfg: "1", steps: "10", resolution: "1248x832", denoise: "1" }
    : { cfg: "7", steps: "20", resolution: "1024x1024", denoise: "1" };
  const selected = state.engineSettings[state.engine] ?? defaults;
  for (const name of ["cfg", "steps", "resolution", "denoise"]) elements[name].value = selected[name] ?? defaults[name];
  updateEngineFields();
  loadInventory({ ...collectSettings(), ...selected });
}

function persistSettings() {
  return storage?.saveSettings(collectSettings()) ?? false;
}

function queueSettingsSave() {
  if (state.settingsSaveTimer !== null) window.clearTimeout(state.settingsSaveTimer);
  state.settingsSaveTimer = window.setTimeout(() => {
    state.settingsSaveTimer = null;
    persistSettings();
  }, SETTINGS_SAVE_DELAY_MS);
}

async function loadInventory(preferredSettings = collectSettings()) {
  const requestId = ++state.inventoryRequest;
  state.ready = false;
  setError();
  setConnection("loading", "Connexion au backend…");
  elements.generateButton.disabled = true;
  elements.model.disabled = true;
  elements.textEncoder.disabled = true;
  elements.method.disabled = true;
  elements.sigmas.disabled = true;

  try {
    if (state.engine === "krea2") {
      const response = await fetch("/api/models/krea2", { cache: "no-store" });
      if (!response.ok) throw new Error(`Découverte Krea v2 indisponible (${response.status}).`);
      const inventory = await response.json();
      if (requestId !== state.inventoryRequest) return;
      if (!Array.isArray(inventory.models) || !inventory.models.length)
        throw new Error("Aucun checkpoint Krea dans krea2/checkpoints. Ajoutez vos poids puis actualisez.");
      if (!Array.isArray(inventory.text_encoders) || !inventory.text_encoders.length)
        throw new Error("Aucun encodeur dans text_encoders/qwen3vl. Ajoutez vos poids puis actualisez.");
      state.inventory = inventory;
      fillSelect(elements.model, inventory.models, preferredSettings.model);
      fillSelect(elements.textEncoder, inventory.text_encoders, preferredSettings.textEncoder);
      fillSelect(elements.method, [{ name: "euler", label: "Euler" }]);
      fillSelect(elements.sigmas, [{ name: "beta", label: "Beta" }]);
      state.ready = true;
      updateEngineFields();
      elements.generateButton.disabled = state.generating;
      persistSettings();
      setConnection("connected", `Krea v2 · ${inventory.models.length} checkpoint(s) disponible(s)`);
      return;
    }
    const response = await fetch("/api/models", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Inventaire indisponible (${response.status}).`);
    }
    const inventory = await response.json();
    if (requestId !== state.inventoryRequest) return;
    if (!Array.isArray(inventory.models) || inventory.models.length === 0) {
      throw new Error("Aucun checkpoint SDXL n’est disponible dans le backend.");
    }
    state.inventory = inventory;
    fillSelect(elements.model, inventory.models, preferredSettings.model);
    fillSelect(elements.method, inventory.sampling_methods, preferredSettings.method);
    refreshSigmaOptions(preferredSettings.sigmas);
    state.ready = true;
    updateEngineFields();
    elements.generateButton.disabled = state.generating;
    persistSettings();
    setConnection("connected", `${inventory.models.length} modèle${inventory.models.length > 1 ? "s" : ""} disponible${inventory.models.length > 1 ? "s" : ""}`);
  } catch (error) {
    if (requestId !== state.inventoryRequest) return;
    state.inventory = null;
    elements.model.replaceChildren(new Option("Backend indisponible", ""));
    elements.textEncoder.replaceChildren(new Option("Encodeurs indisponibles", ""));
    setConnection("error", "Backend indisponible");
    setError(
      `${error.message} Vérifiez que le backend est lancé sur ${elements.backendUrl.value}.`,
    );
  }
}

function updateCount(input, output) {
  output.textContent = String(input.value.length);
}

function updateNegativePromptMode({ focus = false } = {}) {
  const custom = state.engine !== "krea2" && elements.negativeMode.value === "custom";
  elements.negativePromptField.hidden = !custom;
  elements.negativePrompt.disabled = !custom;
  if (custom && focus) elements.negativePrompt.focus();
}

function updateReferencePreview(statusText = "Cliquer pour remplacer") {
  const file = elements.reference.files?.[0];
  if (state.referenceUrl) URL.revokeObjectURL(state.referenceUrl);
  state.referenceUrl = null;
  elements.referencePreview.hidden = true;

  if (!file) {
    elements.uploadTitle.innerHTML = "Ajouter un portrait <span>*</span>";
    elements.uploadHint.textContent = "JPEG, PNG, WebP, BMP ou TIFF · 20 Mio max.";
    elements.clearReference.hidden = true;
    return;
  }

  elements.uploadTitle.textContent = file.name;
  elements.uploadHint.textContent = `${(file.size / 1024 / 1024).toFixed(2)} Mio · ${statusText}`;
  state.referenceUrl = URL.createObjectURL(file);
  elements.referencePreview.src = state.referenceUrl;
  elements.referencePreview.hidden = false;
  elements.clearReference.hidden = false;
}

async function persistReference() {
  const file = elements.reference.files?.[0];
  if (!file) return;
  if (file.size === 0 || file.size > MAX_REFERENCE_BYTES) {
    updateReferencePreview("non valide · non conservée");
    return;
  }
  try {
    await storage.saveReference(file);
    if (elements.reference.files?.[0] === file) {
      updateReferencePreview("conservée localement · cliquer pour remplacer");
    }
  } catch {
    if (elements.reference.files?.[0] === file) {
      updateReferencePreview("non persistée · cliquer pour remplacer");
    }
  }
}

async function restoreReference() {
  try {
    const saved = await storage.loadReference();
    if (!saved?.blob || !(saved.blob instanceof Blob)) return;
    const file = new File([saved.blob], saved.filename || "reference.png", {
      type: saved.contentType || saved.blob.type,
      lastModified: Number(saved.lastModified) || Date.now(),
    });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    elements.reference.files = transfer.files;
    validateReference();
    updateReferencePreview("restaurée localement · cliquer pour remplacer");
  } catch {
    // Le formulaire reste utilisable si la restauration locale est indisponible.
  }
}

async function clearReference() {
  setError();
  try {
    await storage.clearReference();
    elements.reference.value = "";
    elements.reference.setCustomValidity("");
    updateReferencePreview();
  } catch (error) {
    setError(error.message || "Impossible d’effacer la photo conservée localement.");
  }
}

function validateReference() {
  const file = elements.reference.files?.[0];
  if (!file) {
    elements.reference.setCustomValidity("Ajoutez une image de référence.");
  } else if (file.size === 0) {
    elements.reference.setCustomValidity("L’image de référence est vide.");
  } else if (file.size > MAX_REFERENCE_BYTES) {
    elements.reference.setCustomValidity("L’image de référence dépasse 20 Mio.");
  } else {
    elements.reference.setCustomValidity("");
  }
}

function validateSeed() {
  const value = elements.seed.value.trim();
  let message = "";
  if (!value) {
    elements.seed.setCustomValidity("");
    return;
  }
  if (!/^-?\d+$/.test(value)) {
    message = "La seed doit être un entier.";
  } else {
    const seed = BigInt(value);
    if (seed < -1n || seed > MAX_SEED) {
      message = `La seed doit valoir -1, 0, ou être comprise entre 1 et ${MAX_SEED}.`;
    }
  }
  elements.seed.setCustomValidity(message);
}

function updatePromptValidation() {
  const limit = state.engine === "krea2" ? 8000 : 4000;
  elements.prompt.maxLength = limit;
  elements.promptLimit.textContent = String(limit);
  elements.prompt.setCustomValidity(elements.prompt.value.length > limit
    ? `Le prompt dépasse ${limit} caractères pour ce moteur.` : "");
  updateCount(elements.prompt, elements.promptCount);
}

function buildGenerationBody() {
  const form = new FormData();
  if (state.engine === "krea2") {
    const [width, height] = elements.resolution.value.split("x");
    const values = { prompt: elements.prompt.value.trim(), width, height,
      model: elements.model.value, text_encoder: elements.textEncoder.value,
      seed: elements.seed.value.trim(), steps: elements.steps.value, cfg: elements.cfg.value,
      sampler: elements.method.value, scheduler: elements.sigmas.value, denoise: elements.denoise.value };
    for (const [name, value] of Object.entries(values)) if (value !== "") form.append(name, value);
    return form;
  }
  form.append("reference", elements.reference.files[0]);
  form.append("character", elements.character.value.trim());
  form.append("prompt", elements.prompt.value.trim());

  if (elements.negativeMode.value === "custom") {
    form.append("negative_prompt", elements.negativePrompt.value);
  } else if (elements.negativeMode.value === "disabled") {
    form.append("negative_prompt", "");
  }

  form.append("clip_skip_2", String(elements.clipSkip2.checked));
  form.append("model", elements.model.value);
  const [width, height] = elements.resolution.value.split("x");
  form.append("width", width);
  form.append("height", height);
  const optionalValues = {
    cfg: elements.cfg.value,
    steps: elements.steps.value,
    strength: elements.strength.value,
    seed: elements.seed.value.trim(),
  };
  for (const [name, value] of Object.entries(optionalValues)) {
    if (value !== "") form.append(name, value);
  }
  form.append("method", elements.method.value);
  form.append("sigmas", elements.sigmas.value);
  return form;
}

async function responseError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail?.message === "string") return payload.detail.message;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item.msg).filter(Boolean).join(" · ");
    }
  } catch {
    // La réponse n'est pas du JSON exploitable.
  }
  return `Génération impossible (${response.status}).`;
}

function filenameFromResponse(response) {
  const disposition = response.headers.get("Content-Disposition") ?? "";
  return disposition.match(/filename="([^"]+)"/)?.[1] ?? "generation.png";
}

function showGenerating(active) {
  state.generating = active;
  elements.generateButton.disabled = active || !state.ready;
  elements.engine.disabled = active;
  elements.reconnectButton.disabled = active;
  elements.generateButton.classList.toggle("loading", active);
  elements.generateButton.querySelector(".button-label").textContent = active
    ? "Génération en cours"
    : "Générer l’image";
  if (active) {
    elements.emptyResult.hidden = true;
    elements.generatedImage.hidden = true;
    elements.generatingResult.hidden = false;
  }
}

function resultFromResponse(response, blob) {
  return {
    blob,
    filename: filenameFromResponse(response),
    seed: response.headers.get("X-Generation-Seed") ?? "—",
    model: response.headers.has("X-Krea2-Model")
      ? decodeURIComponent(response.headers.get("X-Krea2-Model"))
      : response.headers.get("X-Generation-Model") ?? response.headers.get("X-SDXL-Model") ?? elements.model.value,
    method: response.headers.get("X-Sampling-Method") ?? elements.method.value,
    sigmas: response.headers.get("X-Sigma-Schedule") ?? elements.sigmas.value,
  };
}

function showResult(result) {
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  state.resultUrl = URL.createObjectURL(result.blob);
  elements.generatedImage.src = state.resultUrl;
  elements.generatedImage.hidden = false;
  elements.emptyResult.hidden = true;
  elements.generatingResult.hidden = true;
  elements.resultSeed.textContent = result.seed ?? "—";
  elements.resultModel.textContent = result.model ?? "—";
  elements.resultMethod.textContent = result.method ?? "—";
  elements.resultSigmas.textContent = result.sigmas ?? "—";
  elements.resultMeta.hidden = false;
  elements.downloadButton.href = state.resultUrl;
  elements.downloadButton.download = result.filename || "generation.png";
  elements.downloadButton.hidden = false;
}

function openImageLightbox() {
  if (!state.resultUrl || elements.generatedImage.hidden || elements.imageLightbox.open) return;
  elements.lightboxImage.src = state.resultUrl;
  elements.imageLightbox.showModal();
}

function activateImageLightbox(event) {
  if (event.type === "keydown" && event.key !== "Enter" && event.key !== " ") return;
  event.preventDefault();
  openImageLightbox();
}

function closeImageLightbox() {
  if (elements.imageLightbox.open) elements.imageLightbox.close();
}

async function generate(event) {
  event.preventDefault();
  setError();
  if (state.engine === "sdxl") validateReference();
  validateSeed();
  updatePromptValidation();

  if (!elements.form.checkValidity()) {
    elements.form.reportValidity();
    return;
  }
  if (!state.ready || state.generating) return;

  persistSettings();
  showGenerating(true);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(state.engine === "krea2" ? "/api/generate/krea2" : "/api/generate", {
      method: "POST",
      body: buildGenerationBody(),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await responseError(response));
    const result = resultFromResponse(response, await response.blob());
    showResult(result);
  } catch (error) {
    elements.generatingResult.hidden = true;
    if (!state.resultUrl) elements.emptyResult.hidden = false;
    else elements.generatedImage.hidden = false;
    setError(
      error.name === "AbortError"
        ? "Le backend n’a pas répondu après cinq minutes. La génération a été interrompue côté interface."
        : error.message,
    );
  } finally {
    window.clearTimeout(timeout);
    showGenerating(false);
  }
}

function useDroppedFile(event) {
  event.preventDefault();
  elements.uploadZone.classList.remove("dragging");
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  elements.reference.files = transfer.files;
  validateReference();
  updateReferencePreview();
  persistReference();
}

async function clearLocalData() {
  const confirmed = window.confirm(
    "Cette action supprime de ce navigateur la photo d’identité mémorisée et tous les réglages enregistrés. L’aperçu actuel disparaîtra au rechargement. Aucun modèle, fichier du backend ou PNG déjà téléchargé ne sera supprimé.\n\nVoulez-vous continuer ?",
  );
  if (!confirmed) return;
  setError();
  try {
    await storage.clearAll();
    window.location.reload();
  } catch (error) {
    setError(error.message || "Impossible d’effacer les données locales.");
  }
}

elements.reconnectButton.addEventListener("click", () => loadInventory(collectSettings()));
elements.engine.addEventListener("change", changeEngine);
elements.backendUrl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadInventory(collectSettings());
  }
});
elements.form.addEventListener("submit", generate);
elements.form.addEventListener("input", queueSettingsSave);
elements.form.addEventListener("change", queueSettingsSave);
elements.advancedSettings.addEventListener("toggle", queueSettingsSave);
elements.method.addEventListener("change", () => refreshSigmaOptions());
elements.negativeMode.addEventListener("change", () =>
  updateNegativePromptMode({ focus: true }),
);
elements.reference.addEventListener("change", () => {
  validateReference();
  updateReferencePreview();
  persistReference();
});
elements.clearReference.addEventListener("click", clearReference);
elements.clearLocalData.addEventListener("click", clearLocalData);
elements.generatedImage.addEventListener("click", activateImageLightbox);
elements.generatedImage.addEventListener("keydown", activateImageLightbox);
elements.closeLightbox.addEventListener("click", closeImageLightbox);
elements.imageLightbox.addEventListener("click", (event) => {
  if (event.target === elements.imageLightbox) closeImageLightbox();
});
elements.imageLightbox.addEventListener("close", () => {
  elements.lightboxImage.removeAttribute("src");
});
elements.seed.addEventListener("input", validateSeed);
elements.prompt.addEventListener("input", updatePromptValidation);
elements.negativePrompt.addEventListener("input", () =>
  updateCount(elements.negativePrompt, elements.negativeCount),
);
elements.uploadZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.uploadZone.classList.add("dragging");
});
elements.uploadZone.addEventListener("dragleave", () =>
  elements.uploadZone.classList.remove("dragging"),
);
elements.uploadZone.addEventListener("drop", useDroppedFile);
window.addEventListener("scroll", queueColumnLayoutUpdate, { passive: true });
window.addEventListener("resize", queueColumnLayoutUpdate, { passive: true });
wideLayout.addEventListener("change", queueColumnLayoutUpdate);
window.addEventListener("beforeunload", () => {
  persistSettings();
  if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
  if (state.referenceUrl) URL.revokeObjectURL(state.referenceUrl);
});

updateNegativePromptMode();
updateColumnLayout();

async function initialize() {
  const savedSettings = storage?.loadSettings() ?? null;
  applySettings(savedSettings);
  try {
    const response = await fetch("/frontend-config.json", { cache: "no-store" });
    if (response.ok) {
      const config = await response.json();
      elements.backendUrl.value = config.backend_url;
    }
  } catch {
    // La valeur par défaut reste affichée si la configuration est indisponible.
  }
  await Promise.all([
    loadInventory(savedSettings ?? collectSettings()),
    restoreReference(),
  ]);
}

initialize();
