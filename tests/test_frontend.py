from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from frontend.server import build_server, frontend_directory


class _FormFieldParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.fields: dict[str, dict[str, str | None]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"input", "select", "textarea"}:
            return
        attributes = dict(attrs)
        name = attributes.get("name")
        if name:
            self.fields[name] = attributes


def test_frontend_directory_is_resolved_from_server_module() -> None:
    directory = frontend_directory()
    html = (directory / "index.html").read_text(encoding="utf-8")

    assert directory == Path(__file__).parents[1] / "frontend"
    assert (directory / "index.html").is_file()
    assert (directory / "storage.js").is_file()
    assert html.index('src="storage.js"') < html.index('src="app.js"')


def test_frontend_marks_every_backend_required_field_as_required() -> None:
    parser = _FormFieldParser()
    parser.feed((frontend_directory() / "index.html").read_text(encoding="utf-8"))

    for name in ("reference", "character", "prompt", "model"):
        assert "required" in parser.fields[name]


def test_wide_layout_separates_primary_controls_and_result_cards() -> None:
    html = (frontend_directory() / "index.html").read_text(encoding="utf-8")
    styles = (frontend_directory() / "styles.css").read_text(encoding="utf-8")

    assert 'class="primary-settings card"' in html
    assert 'class="generation-controls card"' in html
    assert 'class="result-panel card"' in html
    assert 'class="card-column primary-column"' in html
    assert 'class="card-column generation-column"' in html
    assert 'class="card-column result-column"' in html
    assert "@media (min-width: 1180px)" in styles
    assert "grid-template-columns:" in styles


def test_wide_layout_scrolls_columns_without_card_scrollbars() -> None:
    styles = (frontend_directory() / "styles.css").read_text(encoding="utf-8")
    source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    wide_layout = styles.split("@media (min-width: 1180px)", 1)[1].split(
        "@media (max-width: 900px)", 1
    )[0]

    assert ".card-column" in wide_layout
    assert ".card-column.is-column-pinned > .card" in wide_layout
    assert "position: fixed" in wide_layout
    assert "padding-bottom: 40px" in wide_layout
    assert "max-height:" not in wide_layout
    assert "overflow-y:" not in wide_layout
    assert "scrollbar" not in wide_layout
    assert ".primary-settings::-webkit-scrollbar" not in wide_layout
    assert "window.matchMedia(WIDE_LAYOUT_QUERY)" in source
    assert 'column.classList.toggle("is-column-pinned"' in source
    assert 'column.addEventListener("wheel", scrollPinnedColumn' in source
    assert "const keepsStickyWhileScrollingUp = delta < 0" in source
    assert "nextOffset === scrollState.offset && !keepsStickyWhileScrollingUp" in source
    assert "event.preventDefault()" in source


def test_sdxl_model_selector_lives_in_advanced_controls() -> None:
    html = (frontend_directory() / "index.html").read_text(encoding="utf-8")
    controls_start = html.index('<section class="generation-controls card">')
    controls_end = html.index("</section>", controls_start)
    model_position = html.index('id="model"')

    assert controls_start < model_position < controls_end


def test_local_data_action_lives_in_advanced_controls_without_footer() -> None:
    html = (frontend_directory() / "index.html").read_text(encoding="utf-8")
    source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    advanced_start = html.index('<div class="advanced-content">')
    advanced_end = html.index("</details>", advanced_start)
    clear_action_position = html.index('id="clearLocalData"')

    assert advanced_start < clear_action_position < advanced_end
    assert "<footer" not in html
    assert "photo d’identité mémorisée" in source
    assert "Aucun modèle, fichier du backend" in source


def test_character_and_clear_reference_share_compact_row() -> None:
    html = (frontend_directory() / "index.html").read_text(encoding="utf-8")
    styles = (frontend_directory() / "styles.css").read_text(encoding="utf-8")
    row_start = html.index('<div class="identity-details">')
    row_end = html.index("</div>", html.index("</div>", row_start) + 1)
    row = html[row_start:row_end]

    assert 'id="character"' in row
    assert 'id="clearReference"' in row
    assert "Oublier la photo" in row
    assert "grid-template-columns: minmax(0, 1fr) max-content" in styles


def test_frontend_client_sends_every_generation_parameter() -> None:
    source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    direct_fields = {
        "reference",
        "character",
        "prompt",
        "negative_prompt",
        "clip_skip_2",
        "model",
        "method",
        "sigmas",
        "width",
        "height",
    }

    assert all(f'form.append("{name}"' in source for name in direct_fields)
    assert all(f"{name}:" in source for name in ("cfg", "steps", "strength", "seed"))
    assert "form.append(name, value)" in source


def test_resolution_selector_offers_optimal_sdxl_sizes_and_persists_choice() -> None:
    html = (frontend_directory() / "index.html").read_text(encoding="utf-8")
    source = (frontend_directory() / "app.js").read_text(encoding="utf-8")

    assert 'id="resolution"' in html
    assert '<option value="1024x1024" selected>' in html
    assert '<optgroup label="Portrait">' in html
    assert '<optgroup label="Paysage">' in html
    for resolution in (
        "896x1152",
        "832x1216",
        "768x1344",
        "640x1536",
        "1152x896",
        "1216x832",
        "1344x768",
        "1536x640",
    ):
        assert f'value="{resolution}"' in html
    assert "resolution: elements.resolution.value" in source
    assert "elements.resolution.value = settings.resolution" in source


def test_frontend_persists_settings_and_reference_but_not_output() -> None:
    app_source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    storage_source = (frontend_directory() / "storage.js").read_text(encoding="utf-8")

    assert "localStorageBackend?.setItem" in storage_source
    assert 'const LAST_REFERENCE_KEY = "last-reference"' in storage_source
    assert "indexedDbBackend.open" in storage_source
    assert "saveReference" in app_source
    assert "restoreReference" in app_source
    assert "saveLastResult" not in app_source
    assert "loadLastResult" not in app_source
    assert "putArtifact(LAST_RESULT" not in storage_source
    assert "const DATABASE_VERSION = 2" in storage_source
    assert 'artifactStore.delete("last-result")' in storage_source


def test_frontend_persists_advanced_settings_disclosure_state() -> None:
    source = (frontend_directory() / "app.js").read_text(encoding="utf-8")

    assert "advancedOpen: elements.advancedSettings.open" in source
    assert 'typeof settings.advancedOpen === "boolean"' in source
    assert "elements.advancedSettings.open = settings.advancedOpen" in source
    assert 'elements.advancedSettings.addEventListener("toggle", queueSettingsSave)' in source


def test_loading_spinners_keep_rotating_with_reduced_motion_enabled() -> None:
    styles = (frontend_directory() / "styles.css").read_text(encoding="utf-8")
    reduced_motion = styles.split("@media (prefers-reduced-motion: reduce)", 1)[1]

    assert ".button-loader" in reduced_motion
    assert "animation-duration: 750ms !important" in reduced_motion
    assert ".generation-spinner" in reduced_motion
    assert "animation-duration: 900ms !important" in reduced_motion
    assert reduced_motion.count("animation-iteration-count: infinite !important") == 2


def test_generated_image_opens_an_accessible_fullscreen_lightbox() -> None:
    html = (frontend_directory() / "index.html").read_text(encoding="utf-8")
    styles = (frontend_directory() / "styles.css").read_text(encoding="utf-8")
    source = (frontend_directory() / "app.js").read_text(encoding="utf-8")

    assert 'aria-controls="imageLightbox"' in html
    assert '<dialog' in html
    assert 'id="imageLightbox"' in html
    assert 'id="closeLightbox"' in html
    assert ".image-lightbox[open]" in styles
    assert "width: 100vw" in styles
    assert "height: 100dvh" in styles
    assert "grid-template-rows: minmax(0, 1fr)" in styles
    assert "max-height: calc(100dvh - 108px)" in styles
    assert "max-height: calc(100dvh - 66px)" in styles
    assert "elements.imageLightbox.showModal()" in source
    assert 'elements.generatedImage.addEventListener("click", activateImageLightbox)' in source
    assert 'elements.generatedImage.addEventListener("keydown", activateImageLightbox)' in source
    assert "event.target === elements.imageLightbox" in source


def test_clearing_only_reference_preserves_settings_and_result() -> None:
    app_source = (frontend_directory() / "app.js").read_text(encoding="utf-8")
    storage_source = (frontend_directory() / "storage.js").read_text(encoding="utf-8")

    clear_reference_block = storage_source.split("function clearReference()", 1)[1].split(
        "}", 1
    )[0]

    assert "storage.clearReference()" in app_source
    assert "deleteArtifact(LAST_REFERENCE_KEY)" in clear_reference_block
    assert "SETTINGS_KEY" not in clear_reference_block


def test_build_server_configures_static_root_and_backend(monkeypatch) -> None:
    captured = {}

    class FakeHttpServer:
        def __init__(self, address, handler) -> None:
            captured["address"] = address
            captured["handler"] = handler

    monkeypatch.setattr("frontend.server.ThreadingHTTPServer", FakeHttpServer)

    server = build_server(port=8888, backend_url="http://127.0.0.1:12693")

    assert isinstance(server, FakeHttpServer)
    assert captured["address"] == ("127.0.0.1", 8888)
    assert captured["handler"].keywords["directory"] == str(frontend_directory())
    assert captured["handler"].keywords["backend"].geturl() == (
        "http://127.0.0.1:12693"
    )


def test_krea_proxy_preserves_png_and_generation_headers(monkeypatch) -> None:
    from io import BytesIO
    from urllib.parse import urlsplit
    from frontend.server import FrontendRequestHandler
    seen = {}
    class BackendResponse:
        status = 200
        reason = "OK"
        def read(self):
            return b"png-response"
        def getheader(self, key):
            return {"Content-Type": "image/png", "X-Generation-Model": "krea2", "X-Generation-Seed": "42"}.get(key)
    class Connection:
        def __init__(self, *args, **kwargs): pass
        def request(self, method, path, **kwargs):
            seen.update(method=method, path=path, **kwargs)
        def getresponse(self): return BackendResponse()
        def close(self): seen["closed"] = True
    monkeypatch.setattr("frontend.server.HTTPConnection", Connection)
    handler = object.__new__(FrontendRequestHandler)
    handler.path = "/api/generate/krea2"
    handler.command = "POST"
    handler.backend = urlsplit("http://127.0.0.1:12693")
    body = b"prompt=hello&steps=10&cfg=1"
    handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/x-www-form-urlencoded"}
    handler.rfile = BytesIO(body)
    handler.wfile = BytesIO()
    response_headers = {}
    handler.send_response = lambda *args: None
    handler.send_header = lambda k, v: response_headers.update({k: v})
    handler.end_headers = lambda: None
    handler.do_POST()
    assert seen["path"] == "/generate/krea2"
    assert seen["body"] == body
    assert seen["closed"]
    assert handler.wfile.getvalue() == b"png-response"
    assert response_headers["X-Generation-Model"] == "krea2"
    assert response_headers["X-Generation-Seed"] == "42"


def test_frontend_exposes_krea_fields_without_identity_payload() -> None:
    html = (frontend_directory() / "index.html").read_text()
    source = (frontend_directory() / "app.js").read_text()
    parser = _FormFieldParser()
    parser.feed(html)
    assert 'value="krea2"' in html
    assert parser.fields["denoise"]["value"] == "1"
    krea_body = source.split('function buildGenerationBody()', 1)[1].split('return form;', 1)[0]
    assert "sampler:" in krea_body and "scheduler:" in krea_body and "denoise:" in krea_body
    assert 'form.append("reference"' not in krea_body
    assert 'form.append("character"' not in krea_body
    assert 'fetch("/api/capabilities"' in source
