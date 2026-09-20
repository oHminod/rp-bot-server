// Exerce le vrai client avec un DOM minimal et un backend simulé, sans dépendance.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  value = "";
  options = [];
  disabled = false;
  classList = { toggle() {}, remove() {} };
  addEventListener() {}
  setCustomValidity() {}
  replaceChildren(...children) { this.options = children; }
  append(child) { this.options.push(child); }
}

const nodes = new Map();
const requests = [];
let saved;
let inventory;
const context = vm.createContext({
  FormData, Headers,
  Option: Element,
  PuLIDStorage: { saveSettings(value) { saved = JSON.parse(JSON.stringify(value)); } },
  window: {
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    addEventListener() {}, requestAnimationFrame() { return 1; },
  },
  document: {
    querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, new Element());
      return nodes.get(selector);
    },
    querySelectorAll: () => [],
    createElement: () => new Element(),
  },
  fetch: async (url) => {
    requests.push(url);
    return { ok: true, json: async () => inventory };
  },
});
vm.runInContext(fs.readFileSync("frontend/app.js", "utf8").replace(/initialize\(\);\s*$/, ""), context);
const client = vm.runInContext("({ state, elements, loadInventory, buildGenerationBody, changeEngine, resultFromResponse })", context);
const option = (name, preferred = false) => ({ name, filename: name, default: preferred });

async function main() {
  const { state, elements } = client;
  state.engine = "krea2";
  elements.prompt.value = "photo";
  elements.resolution.value = "1248x832";
  inventory = { models: [option("a.safetensors", true), option("Krea été.SAFETENSORS")],
    text_encoders: [option("Qwen FP8.safetensors", true), option("Qwen BF16.safetensors")] };
  await client.loadInventory({ model: "Krea été.SAFETENSORS", textEncoder: "Qwen BF16.safetensors" });
  assert.equal(requests.at(-1), "/api/models/krea2");
  assert.equal(state.ready, true);
  assert.equal(elements.model.disabled, false);
  assert.equal(elements.textEncoder.disabled, false);
  assert.equal(elements.model.options.length, 2);
  assert.equal(saved.engineSettings.krea2.model, "Krea été.SAFETENSORS");
  assert.equal(saved.engineSettings.krea2.textEncoder, "Qwen BF16.safetensors");
  const body = client.buildGenerationBody();
  assert.equal(body.get("model"), "Krea été.SAFETENSORS");
  assert.equal(body.get("text_encoder"), "Qwen BF16.safetensors");
  assert.equal(body.has("reference"), false);
  assert.equal(client.resultFromResponse({ headers: new Headers({
    "X-Krea2-Model": encodeURIComponent("Krea été.SAFETENSORS"),
  }) }, null).model, "Krea été.SAFETENSORS");

  // Passer à SDXL puis revenir conserve les choix Krea séparément.
  const kreaInventory = inventory;
  inventory = { models: [option("sdxl", true)], sampling_methods: [option("default")],
    sigma_schedules: [option("normal")] };
  elements.engine.value = "sdxl";
  client.changeEngine();
  await new Promise(setImmediate);
  assert.equal(elements.model.value, "sdxl");
  assert.equal(elements.textEncoder.disabled, true);
  inventory = kreaInventory;
  elements.engine.value = "krea2";
  client.changeEngine();
  await new Promise(setImmediate);
  assert.equal(elements.model.value, "Krea été.SAFETENSORS");
  assert.equal(elements.textEncoder.value, "Qwen BF16.safetensors");

  await client.loadInventory({ model: "retiré.safetensors", textEncoder: "retiré.safetensors" });
  assert.equal(elements.model.value, "a.safetensors");
  assert.equal(elements.textEncoder.value, "Qwen FP8.safetensors");
  inventory = { models: kreaInventory.models, text_encoders: [] };
  await client.loadInventory();
  assert.equal(state.ready, false);
  assert.equal(elements.generateButton.disabled, true);
  assert.equal(elements.textEncoder.disabled, true);
  assert.match(elements.formError.textContent, /Aucun encodeur/);
  console.log("Frontend Krea : catalogue, sélection, formulaire, persistance et absence de poids OK.");
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
