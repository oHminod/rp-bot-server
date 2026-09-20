from __future__ import annotations

from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import yaml

from pulid_app.config import ConfigError, load_config
from pulid_app.exceptions import GenerationError, ModelLoadError, ModelNotFoundError
from pulid_app.models.krea2 import (
    inspect_weight_format, load_safetensors_module, text_encoder_key,
    transformer_key, validate_krea2_assets,
)
from pulid_app.paths import configure_external_model_caches
from pulid_app.pipeline.krea2 import Krea2Generator, Krea2Parameters, beta_sigmas, shifted_sigma
from pulid_app.server import create_app
from test_server import _write_config, _request, _generate_request, _image_bytes, FakeEmbeddingModel, FakeMemoryGenerator
from test_krea2_catalog import write_model


def _fake_krea_weights(config_path):
    config = load_config(config_path).krea2
    write_model(config.checkpoint)
    write_model(config.text_encoder)


class FakeKreaGenerator:
    instances = []
    failure = None

    def __init__(self, config, **options):
        self.config = config
        self.options = options
        self.parameters = None
        self.closed = False
        self.close_calls = 0
        self.instances.append(self)

    def generate(self, parameters):
        self.parameters = parameters
        if self.failure:
            raise self.failure
        return Image.new("RGB", (parameters.width, parameters.height), "navy"), {}

    def close(self):
        self.close_calls += 1
        self.closed = True


@pytest.fixture
def app(tmp_path):
    config, _ = _write_config(tmp_path)
    _fake_krea_weights(config)
    FakeKreaGenerator.instances.clear()
    FakeKreaGenerator.failure = None
    FakeMemoryGenerator.instances.clear()
    FakeEmbeddingModel.instances.clear()
    return create_app(config, embedding_memory_mode="cpu", random_seed=lambda: 123,
                      generator_factory=FakeMemoryGenerator,
                      embedding_model_factory=FakeEmbeddingModel,
                      krea2_generator_factory=FakeKreaGenerator,
                      cors_origins=["http://rp-bot.local"])


@pytest.fixture
def cuda_app(app):
    return create_app(app.state.krea2_service.config.source_path, device="cuda",
        generator_factory=FakeMemoryGenerator, embedding_model_factory=FakeEmbeddingModel,
        krea2_generator_factory=FakeKreaGenerator)


@pytest.mark.parametrize("override,expected", [(None, "none"), ("model_cpu_offload", "model_cpu_offload")])
def test_krea_offload_override_is_independent_from_sdxl(app, override, expected):
    application = create_app(app.state.krea2_service.config.source_path, device="cuda",
        offload_strategy="none", krea2_offload_strategy=override,
        generator_factory=FakeMemoryGenerator, embedding_model_factory=FakeEmbeddingModel,
        krea2_generator_factory=FakeKreaGenerator)
    assert application.state.generation_service.offload_strategy == "none"
    assert _request(application, "POST", "/generate/krea2", data={"prompt": "photo"}).status_code == 200
    assert FakeKreaGenerator.instances[-1].options["offload_strategy"] == expected


def test_cuda_krea_reuses_pipeline_and_keeps_png_in_memory(cuda_app, tmp_path):
    before = sorted(tmp_path.rglob("*"))
    assert _request(cuda_app, "POST", "/generate/krea2", data={"prompt": "photo", "seed": 42}).status_code == 200
    generator = FakeKreaGenerator.instances[0]
    assert not generator.closed and generator.options["keep_loaded"]
    assert _request(cuda_app, "GET", "/models").status_code == 200
    assert _request(cuda_app, "POST", "/generate/krea2", data={"prompt": "", "steps": 0}).status_code == 422
    response = _request(cuda_app, "POST", "/generate/krea2",
        data={"prompt": "autre photo", "seed": 43, "width": 64, "height": 80})
    assert response.status_code == 200 and response.headers["x-generation-seed"] == "43"
    assert Image.open(BytesIO(response.content)).size == (64, 80)
    assert len(FakeKreaGenerator.instances) == 1 and not generator.closed
    assert generator.parameters.prompt == "autre photo"
    assert sorted(tmp_path.rglob("*")) == before
    cuda_app.state.krea2_service.close()
    cuda_app.state.krea2_service.close()
    assert generator.close_calls == 1


def test_cuda_krea_failure_evicts_pipeline_and_allows_retry(cuda_app):
    assert _request(cuda_app, "POST", "/generate/krea2", data={"prompt": "photo"}).status_code == 200
    generator = FakeKreaGenerator.instances[0]
    generator.failure = GenerationError("CUDA OOM")
    response = _request(cuda_app, "POST", "/generate/krea2", data={"prompt": "photo"})
    assert response.status_code == 500
    assert response.json()["detail"]["message"] == "CUDA OOM"
    assert generator.close_calls == 1
    assert _request(cuda_app, "POST", "/generate/krea2", data={"prompt": "photo"}).status_code == 200
    assert len(FakeKreaGenerator.instances) == 2
    assert not FakeKreaGenerator.instances[-1].closed


def test_krea_catalog_selection_and_cuda_switching(cuda_app, tmp_path):
    from urllib.parse import unquote
    service = cuda_app.state.krea2_service
    original_config = service.config.krea2
    original_config.checkpoint.unlink()
    original_config.text_encoder.unlink()
    checkpoints = [write_model(original_config.checkpoint.parent / name) for name in
                   ("mon modèle FP8.safetensors", "autre.SAFETENSORS")]
    encoders = [write_model(original_config.text_encoder.parent / name) for name in
                ("Qwen personnalisé.safetensors", "Qwen NVFP4.safetensors")]
    before = sorted(tmp_path.rglob("*"))
    catalog = _request(cuda_app, "GET", "/models/krea2")
    assert catalog.status_code == 200 and catalog.headers["cache-control"] == "no-store"
    assert {item["name"] for item in catalog.json()["models"]} == {path.name for path in checkpoints}
    assert {item["name"] for item in catalog.json()["text_encoders"]} == {path.name for path in encoders}
    assert not FakeKreaGenerator.instances
    for index, (checkpoint, encoder) in enumerate([
        (checkpoints[0], encoders[0]), (checkpoints[0], encoders[0]),
        (checkpoints[1], encoders[0]), (checkpoints[1], encoders[1]),
    ]):
        response = _request(cuda_app, "POST", "/generate/krea2", data={
            "prompt": "image", "width": 64, "height": 64,
            "model": checkpoint.name, "text_encoder": encoder.name})
        assert response.status_code == 200, response.text
        assert response.headers["x-generation-model"] == "krea2"
        assert unquote(response.headers["x-krea2-model"]) == checkpoint.name
        assert unquote(response.headers["x-krea2-text-encoder"]) == encoder.name
        assert len(FakeKreaGenerator.instances) == [1, 1, 2, 3][index]
        active = FakeKreaGenerator.instances[-1]
        assert active.config.krea2.checkpoint == checkpoint
        assert active.config.krea2.text_encoder == encoder
        assert active.config.krea2.text_encoder_config_dir == original_config.text_encoder_config_dir
        assert all(instance.closed for instance in FakeKreaGenerator.instances[:-1])
        assert not active.closed
    for field in ("model", "text_encoder"):
        response = _request(cuda_app, "POST", "/generate/krea2",
                            data={"prompt": "image", field: "missing.safetensors"})
        assert response.status_code == 422
        assert not active.closed
    assert service.config.krea2 == original_config
    assert sorted(tmp_path.rglob("*")) == before


def test_krea_default_uses_catalog_without_renaming_and_headers_support_cors(app):
    config = app.state.krea2_service.config.krea2
    config.checkpoint.unlink()
    config.text_encoder.unlink()
    checkpoint = write_model(config.checkpoint.parent / "Turbo FP8.safetensors")
    encoder = write_model(config.text_encoder.parent / "Qwen FP8.safetensors")
    response = _request(app, "POST", "/generate/krea2", data={"prompt": "image"},
                        headers={"Origin": "http://rp-bot.local"})
    assert response.status_code == 200
    assert FakeKreaGenerator.instances[-1].config.krea2.checkpoint == checkpoint
    assert FakeKreaGenerator.instances[-1].config.krea2.text_encoder == encoder
    assert "X-Krea2-Model" in response.headers["access-control-expose-headers"]
    assert "X-Krea2-Text-Encoder" in response.headers["access-control-expose-headers"]


def test_cuda_krea_closed_on_server_shutdown(cuda_app):
    import asyncio
    import httpx
    async def run():
        async with cuda_app.router.lifespan_context(cuda_app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=cuda_app), base_url="http://testserver") as client:
                assert (await client.post("/generate/krea2", data={"prompt": "photo"})).status_code == 200
            assert not FakeKreaGenerator.instances[0].closed
        assert FakeKreaGenerator.instances[0].close_calls == 1
    asyncio.run(run())


@pytest.mark.parametrize("consumer", ["sdxl", "bge_gpu", "bge_cpu"])
def test_krea_is_evicted_before_another_gpu_model_loads(app, consumer):
    class CheckedSDXL(FakeMemoryGenerator):
        def __init__(self, *args, **kwargs):
            assert FakeKreaGenerator.instances[-1].closed
            super().__init__(*args, **kwargs)
    class CheckedEmbedding(FakeEmbeddingModel):
        def __init__(self, *args, **kwargs):
            assert FakeKreaGenerator.instances[-1].closed == (consumer != "bge_cpu")
            super().__init__(*args, **kwargs)
    application = create_app(app.state.krea2_service.config.source_path, device="cuda",
        embedding_memory_mode="cpu" if consumer == "bge_cpu" else "concurrent",
        generator_factory=CheckedSDXL, embedding_model_factory=CheckedEmbedding,
        krea2_generator_factory=FakeKreaGenerator)
    assert _request(application, "POST", "/generate/krea2", data={"prompt": "photo"}).status_code == 200
    if consumer == "sdxl":
        response = _generate_request(application)
        other = FakeMemoryGenerator.instances[-1]
    else:
        response = _request(application, "POST", "/v1/embeddings",
            json={"model": "text-embedding-bge-m3", "input": "bonjour"})
        other = FakeEmbeddingModel.instances[-1]
    assert response.status_code == 200, response.text
    assert FakeKreaGenerator.instances[0].closed == (consumer != "bge_cpu")
    assert _request(application, "POST", "/generate/krea2", data={"prompt": "retour à Krea"}).status_code == 200
    assert other.closed
    assert len(FakeKreaGenerator.instances) == (1 if consumer == "bge_cpu" else 2)


def test_simultaneous_sdxl_and_bge_wait_for_krea_eviction(app):
    import asyncio
    import threading
    import httpx
    entered, release = threading.Event(), threading.Event()
    class SlowCloseKrea(FakeKreaGenerator):
        def close(self):
            entered.set()
            assert release.wait(5)
            super().close()
    class CheckedSDXL(FakeMemoryGenerator):
        def __init__(self, *args, **kwargs):
            assert FakeKreaGenerator.instances[0].closed
            super().__init__(*args, **kwargs)
    class CheckedEmbedding(FakeEmbeddingModel):
        def __init__(self, *args, **kwargs):
            assert FakeKreaGenerator.instances[0].closed
            super().__init__(*args, **kwargs)
    application = create_app(app.state.krea2_service.config.source_path, device="cuda",
        generator_factory=CheckedSDXL, embedding_model_factory=CheckedEmbedding,
        krea2_generator_factory=SlowCloseKrea)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
            assert (await client.post("/generate/krea2", data={"prompt": "photo"})).status_code == 200
            sdxl = asyncio.create_task(client.post("/generate",
                files={"reference": ("face.png", _image_bytes(), "image/png")},
                data={"character": "test", "prompt": "photo", "model": "realvisxl"}))
            assert await asyncio.to_thread(entered.wait, 5)
            bge = asyncio.create_task(client.post("/v1/embeddings",
                json={"model": "text-embedding-bge-m3", "input": "bonjour"}))
            try:
                await asyncio.sleep(.05)
                assert not sdxl.done() and not bge.done()
                assert not FakeMemoryGenerator.instances and not FakeEmbeddingModel.instances
            finally:
                release.set()
            assert all(response.status_code == 200 for response in await asyncio.gather(sdxl, bge))
            assert FakeKreaGenerator.instances[0].close_calls == 1
    asyncio.run(run())


def test_http_defaults_png_no_identity_and_no_disk_writes(app, tmp_path):
    before = sorted(tmp_path.rglob("*"))
    response = _request(app, "POST", "/generate/krea2", data={"prompt": "une plage"}, headers={"Origin": "http://rp-bot.local"})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-generation-seed"] == "123"
    assert response.headers["x-generation-model"] == "krea2"
    assert response.headers["x-sampling-method"] == "euler"
    assert response.headers["x-sigma-schedule"] == "beta"
    assert response.headers["cache-control"] == "no-store"
    assert "X-Generation-Model" in response.headers["access-control-expose-headers"]
    assert Image.open(BytesIO(response.content)).size == (1248, 832)
    assert FakeKreaGenerator.instances[0].parameters == Krea2Parameters("une plage", seed=123)
    assert FakeKreaGenerator.instances[0].closed
    assert not FakeMemoryGenerator.instances
    assert not FakeEmbeddingModel.instances
    assert sorted(tmp_path.rglob("*")) == before


def test_http_explicit_parameters_and_multipart(app):
    response = _request(app, "POST", "/generate/krea2", files={"prompt": (None, "portrait")},
                        data={"width": "64", "height": "80", "steps": "3", "cfg": "0.5", "denoise": "0.4", "seed": "42"})
    assert response.status_code == 200, response.text
    p = FakeKreaGenerator.instances[-1].parameters
    assert (p.width, p.height, p.steps, p.cfg, p.denoise, p.seed) == (64, 80, 3, 0.5, 0.4, 42)


@pytest.mark.parametrize("length,status", [(4001, 200), (8000, 200), (8001, 422)])
def test_krea_extended_prompt_character_limit(app, length, status):
    prompt = "x" * length
    response = _request(app, "POST", "/generate/krea2",
        data={"prompt": prompt, "width": 64, "height": 64})
    assert response.status_code == status
    if status == 200:
        assert FakeKreaGenerator.instances[-1].parameters.prompt == prompt
        assert Krea2Parameters(prompt).prompt == prompt
    else:
        assert not FakeKreaGenerator.instances
        with pytest.raises(ValueError, match="8000"):
            Krea2Parameters(prompt)


def test_diffusion_consumes_text_tokens_after_position_512(tmp_path):
    """La seconde moitié du conditionnement 1024 change le calcul du DiT."""
    configure_external_model_caches(tmp_path)
    import torch
    from diffusers import Krea2Pipeline, Krea2Transformer2DModel
    from pulid_app.pipeline.krea2 import KREA2_TEXT_SEQUENCE_LENGTH

    torch.manual_seed(18)
    transformer = Krea2Transformer2DModel(
        num_layers=1, attention_head_dim=8, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, timestep_embed_dim=16, text_hidden_dim=16, num_text_layers=2,
        text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=32,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(2, 2, 4),
    ).eval()
    assert KREA2_TEXT_SEQUENCE_LENGTH == 1024
    context = torch.randn(1, KREA2_TEXT_SEQUENCE_LENGTH, 2, 16)
    changed = context.clone()
    changed[:, 512:] = torch.randn_like(changed[:, 512:])
    mask = torch.ones(1, KREA2_TEXT_SEQUENCE_LENGTH, dtype=torch.bool)
    kwargs = dict(hidden_states=torch.randn(1, 4, 64), timestep=torch.tensor([.5]),
        position_ids=Krea2Pipeline.prepare_position_ids(KREA2_TEXT_SEQUENCE_LENGTH, 2, 2, torch.device("cpu")),
        encoder_attention_mask=mask, return_dict=False)
    with torch.inference_mode():
        before = transformer(encoder_hidden_states=context, **kwargs)[0]
        after = transformer(encoder_hidden_states=changed, **kwargs)[0]
        assert torch.isfinite(after).all() and not torch.allclose(before, after)
        # Quand ces mêmes positions sont masquées, elles cessent d'influencer l'image.
        mask[:, 512:] = False
        masked = transformer(encoder_hidden_states=context, **kwargs)[0]
        masked_changed = transformer(encoder_hidden_states=changed, **kwargs)[0]
        torch.testing.assert_close(masked, masked_changed)


@pytest.mark.parametrize("invalid", [
    {"prompt": "   "}, {"width": "65"}, {"width": "2064"}, {"height": "8"},
    {"steps": "0"}, {"steps": "201"}, {"steps": "2.5"}, {"cfg": "nan"},
    {"cfg": "inf"}, {"cfg": "-1"}, {"seed": "-2"}, {"seed": str(2**63)},
    {"denoise": "0"}, {"denoise": "nan"}, {"denoise": "1.1"},
    {"sampler": "dpmpp_2m"}, {"scheduler": "karras"}, {"reference": "photo"},
    {"character": "alice"}, {"strength": "0.8"}, {"model": "../../checkpoint"},
])
def test_http_validation_happens_before_model_allocation(app, invalid):
    response = _request(app, "POST", "/generate/krea2", data={"prompt": "photo", **invalid})
    assert response.status_code == 422
    assert not FakeKreaGenerator.instances


@pytest.mark.parametrize("error,status", [
    (ModelNotFoundError("Absent : /models/krea2.safetensors"), 422),
    (ModelLoadError("Poids invalides : /models/krea2.safetensors"), 500),
    (RuntimeError("OOM"), 500),
    (GenerationError("Attention Krea CUDA fusionnée indisponible ; scripts/benchmark_krea2_cuda.py --attention"), 500),
])
def test_http_errors_cleanup_and_recovery(app, error, status):
    FakeKreaGenerator.failure = error
    try:
        response = _request(app, "POST", "/generate/krea2", data={"prompt": "photo"})
        assert response.status_code == status
        assert response.json()["detail"]["message"] == str(error)
        assert response.json()["detail"]["error"] == type(error).__name__
        assert FakeKreaGenerator.instances[-1].closed
    finally:
        FakeKreaGenerator.failure = None
    assert _request(app, "POST", "/generate/krea2", data={"prompt": "photo"}).status_code == 200


def test_config_defaults_env_and_escape_rejected(tmp_path, monkeypatch):
    path, models = _write_config(tmp_path)
    config = load_config(path)
    assert config.krea2.checkpoint == models / "krea2/checkpoints/krea2.safetensors"
    monkeypatch.setenv("PULID_KREA2_TEXT_ENCODER", "text_encoders/qwen3vl/custom.safetensors")
    assert load_config(path).krea2.text_encoder == models / "text_encoders/qwen3vl/custom.safetensors"
    monkeypatch.setenv("PULID_KREA2_CHECKPOINT", "../outside.safetensors")
    with pytest.raises(ConfigError, match="doit rester sous"):
        load_config(path)


def test_missing_assets_are_actionable(tmp_path):
    path, _ = _write_config(tmp_path)
    config = load_config(path)
    with pytest.raises(ModelNotFoundError, match="manuellement") as error:
        validate_krea2_assets(config.krea2)
    assert str(config.krea2.checkpoint) in str(error.value)


def test_beta_schedule_matches_workflow_and_partial_denoise():
    import numpy as np
    # Indices du scheduler beta de référence, sur ModelSamplingFlux (10000 temps).
    assert beta_sigmas(10, 1) == pytest.approx([1, .9608, .8779, .7672, .6371, .5, .363, .2329, .1222, .0393], abs=0.003)
    full = beta_sigmas(20, 1)
    assert beta_sigmas(10, .5) == full[-10:]
    assert shifted_sigma(1) == 1
    assert 0 < shifted_sigma(full[-10]) < 1
    assert len(beta_sigmas(1, 1)) == 1
    sigmas = beta_sigmas(200, .01)
    assert len(sigmas) <= 200
    assert np.all(np.diff(sigmas) < 0)


def test_native_key_mapping():
    assert transformer_key("model.diffusion_model.blocks.0.mod.lin") == "transformer_blocks.0.scale_shift_table"
    assert transformer_key("txtfusion.refiner_blocks.1.attn.qknorm.qnorm.scale") == "text_fusion.refiner_blocks.1.attn.norm_q.weight"
    assert transformer_key("blocks.2.mlp.gate.weight") == "transformer_blocks.2.ff.gate.weight"
    assert transformer_key("last.modulation.lin") == "final_layer.scale_shift_table"
    assert text_encoder_key("model.language_model.layers.0.self_attn.q_proj.weight") == "layers.0.self_attn.q_proj.weight"
    assert text_encoder_key("model.visual.patch_embed.weight") is None
    assert text_encoder_key("model.layers.0.mlp.gate_proj.weight") == "layers.0.mlp.gate_proj.weight"


def test_strict_streaming_loader_and_quantized_rejection(tmp_path):
    configure_external_model_caches(tmp_path)
    import torch
    from safetensors.torch import save_file
    with torch.device("meta"):
        module = torch.nn.Linear(3, 2)
    path = tmp_path / "tiny.safetensors"
    save_file({"weight": torch.ones(2, 3), "bias": torch.zeros(2)}, str(path))
    inspect_weight_format(path)
    loaded = load_safetensors_module(module, path, lambda key: key, dtype=torch.float32)
    assert torch.equal(loaded(torch.ones(1, 3)), torch.full((1, 2), 3.0))
    save_file({"weight": torch.ones(2, 3)}, str(path))
    with pytest.raises(ModelLoadError, match="incomplets"):
        load_safetensors_module(module, path, lambda key: key, dtype=torch.float32)
    save_file({"weight": torch.ones(2, 3, dtype=torch.int8)}, str(path))
    with pytest.raises(ModelLoadError, match="quantifiés"):
        inspect_weight_format(path)


def test_generator_maps_cfg_and_scales_partial_noise(tmp_path, monkeypatch):
    path, _ = _write_config(tmp_path)
    config = load_config(path)
    configure_external_model_caches(config.models_root)
    import torch
    calls = []
    class Pipeline:
        prompt_template_encode_prefix = "system "
        prompt_template_encode_start_idx = 1
        prompt_template_encode_num_suffix_tokens = 5
        def tokenizer(self, text, **kwargs):
            return {"input_ids": list(range(min(len(text.split()), kwargs["max_length"])))}
        def prepare_latents(self, *args):
            return torch.ones((1, 4, 64))
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (64, 64))])
        def to(self, device, **kwargs):
            assert device == "cpu" and kwargs == {"silence_dtype_warnings": True}
            return self
    monkeypatch.setattr("pulid_app.pipeline.krea2.load_krea2_pipeline", lambda *a, **kw: Pipeline())
    generator = Krea2Generator(config)
    try:
        image, metadata = generator.generate(Krea2Parameters("photo", width=64, height=64, seed=42, cfg=.5, denoise=.5))
        assert image.size == (64, 64)
        assert metadata["identity_transfer"] is False
        assert calls[0]["guidance_scale"] == -.5
        assert calls[0]["sigmas"] == beta_sigmas(10, .5)
        assert torch.allclose(calls[0]["latents"], torch.full((1, 4, 64), shifted_sigma(beta_sigmas(10, .5)[0])))
        assert calls[0]["negative_prompt"] == ""
        assert calls[0]["max_sequence_length"] == 6  # photo + suffixe de 5 tokens
        generator.generate(Krea2Parameters("mot " * 600, width=64, height=64, seed=42))
        assert calls[1]["max_sequence_length"] == 605  # Recalcul sur le pipeline déjà chargé.
    finally:
        generator.close()


def test_krea_installation_prepares_config_and_only_downloads_missing_vae(tmp_path, monkeypatch):
    import pulid_app.installer as installer
    from rich.console import Console
    path, models = _write_config(tmp_path)
    config = load_config(path).krea2
    downloads = []
    preparations = []
    def download(root, asset, console):
        downloads.append(asset)
        config.vae.write_bytes(b"vae")
    monkeypatch.setattr(installer, "ensure_huggingface_asset", download)
    monkeypatch.setattr(installer, "ensure_qwen3vl_configuration",
                        lambda root, destination, console: preparations.append((root, destination)))
    monkeypatch.setattr(installer, "_matches", lambda path, digest: path.read_bytes() == b"vae")
    installer.prepare_krea2_assets(models, config, Console(quiet=True))
    installer.prepare_krea2_assets(models, config, Console(quiet=True))
    assert downloads == [installer.KREA2_VAE]
    assert preparations == [(models, config.text_encoder_config_dir)] * 2
    assert config.text_encoder_config_dir.is_dir()
    assert config.checkpoint.parent.is_dir()
    assert not config.checkpoint.exists()
    assert not config.text_encoder.exists()
    config.vae.write_bytes(b"corrupt")
    with pytest.raises(installer.InstallerError, match="aucun fichier existant"):
        installer.prepare_krea2_assets(models, config, Console(quiet=True))
    assert config.vae.read_bytes() == b"corrupt"
    assert len(downloads) == 1


def test_krea_only_install_preserves_existing_configuration(tmp_path, monkeypatch):
    import pulid_app.installer as installer
    from rich.console import Console
    path, models = _write_config(tmp_path)
    before = yaml.safe_load(path.read_text())
    seen = []
    monkeypatch.setattr(installer, "confirm_antelope_license", lambda *a, **kw: pytest.fail("Unrelated identity installation"))
    monkeypatch.setattr(installer, "prepare_required_assets", lambda *a, **kw: pytest.fail("Unrelated downloads"))
    writer = installer.write_krea2_config
    monkeypatch.setattr(installer, "write_krea2_config", lambda root: writer(root, destination=path))
    monkeypatch.setattr(installer, "prepare_krea2_assets", lambda root, config, console: seen.append((root, config)))
    args = installer.build_parser().parse_args(["--models-root", str(models), "--krea2-only"])
    assert installer.run_installation(args, Console(quiet=True)) == 0
    after = yaml.safe_load(path.read_text())
    assert after.pop("krea2")["text_encoder"] == "text_encoders/qwen3vl/qwen3vl_4b_bf16.safetensors"
    assert after == before
    assert seen[0][0] == models


def test_krea_waits_for_bge_even_in_concurrent_cuda_mode(tmp_path):
    import asyncio
    import threading
    import httpx
    path, _ = _write_config(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    running = threading.Event()
    class SlowEmbedding(FakeEmbeddingModel):
        def create_embedding(self, inputs):
            running.set()
            entered.set()
            assert release.wait(5)
            running.clear()
            return super().create_embedding(inputs)
        def close(self):
            assert not running.is_set(), "Cannot close BGE during embedding"
            super().close()
    class CheckedKrea(FakeKreaGenerator):
        def generate(self, parameters):
            assert not running.is_set()
            return super().generate(parameters)
    app = create_app(path, device="cuda", embedding_model_factory=SlowEmbedding,
                     generator_factory=FakeMemoryGenerator, krea2_generator_factory=CheckedKrea)
    _fake_krea_weights(path)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            embed = asyncio.create_task(client.post("/v1/embeddings", json={"model": "text-embedding-bge-m3", "input": "hello"}))
            assert await asyncio.to_thread(entered.wait, 5)
            krea = asyncio.create_task(client.post("/generate/krea2", data={"prompt": "photo"}))
            try:
                await asyncio.sleep(.05)
                assert not krea.done()
            finally:
                release.set()
            assert (await embed).status_code == 200
            assert (await krea).status_code == 200
    asyncio.run(run())


@pytest.mark.parametrize("cfg", [0, .5, 1, 2])
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
@pytest.mark.parametrize("retain_device", [None, "cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_real_diffusers_components_generate_with_tiny_random_weights(tmp_path, monkeypatch, cfg, dtype_name, retain_device):
    """Exerce réellement le débruitage et le VAE, sans poids téléchargés."""
    configure_external_model_caches(tmp_path)
    import torch
    if retain_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requis pour vérifier la résidence du pipeline entre deux images")
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
    from pulid_app.models.krea2 import workflow_pipeline
    from safetensors.torch import save_file
    transformer = Krea2Transformer2DModel(
        num_layers=1, attention_head_dim=8, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, timestep_embed_dim=16, text_hidden_dim=16, num_text_layers=2,
        text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=32,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(2, 2, 4),
    )
    checkpoint = tmp_path / "tiny.safetensors"
    save_file(transformer.state_dict(), str(checkpoint))
    with torch.device("meta"):
        loaded = Krea2Transformer2DModel.from_config(transformer.config)
    dtype = getattr(torch, dtype_name)
    transformer = load_safetensors_module(loaded, checkpoint, transformer_key, dtype=dtype, keep_norm_fp32=True)
    text = Qwen3VLTextModel(Qwen3VLTextConfig(hidden_size=16, intermediate_size=32,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, num_hidden_layers=2, vocab_size=16))
    vae = AutoencoderKLQwenImage(base_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1)
    pipe = workflow_pipeline(transformer=transformer, text_encoder=text, tokenizer=None,
        vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        is_distilled=True, text_encoder_select_layers=(1, 2))
    pipe.set_progress_bar_config(disable=True)
    resets = []
    if retain_device:
        # CPU exerce les vrais hooks Accelerate en CI, CUDA vérifie en plus
        # la résidence effective du débruiteur avec le VAE entre les requêtes.
        from pulid_app.models.krea2_memory import retain_krea2_components
        pipe.enable_model_cpu_offload(device=retain_device)
        pipe.retain_model_hooks = True
        if retain_device == "cpu":
            pipe._offload_device = torch.device("cpu")
            monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (16 * 1024**3, 16 * 1024**3))
            monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
            monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
        retain_krea2_components(pipe, dtype=dtype)
        hooks = tuple(pipe._all_hooks)
        def unexpected_reset(*args, **kwargs):
            pytest.fail("L'offload ne doit pas être réinstallé après chaque image")
        monkeypatch.setattr(pipe, "enable_model_cpu_offload", unexpected_reset)
        monkeypatch.setattr(transformer, "_reset_stateful_cache", lambda: resets.append(True), raising=False)
    embeds = torch.randn(1, 4, 2, 16, dtype=dtype)
    mask = torch.ones(1, 4, dtype=torch.bool)
    try:
        with torch.inference_mode():
            for iteration in range(2 if retain_device else 1):
                if iteration:
                    pipe.prepare_for_next_generation()
                    assert all(p.device.type == retain_device for p in vae.parameters())
                    assert all(p.device.type == retain_device for p in transformer.parameters())
                result = pipe(prompt_embeds=embeds, prompt_embeds_mask=mask,
                    negative_prompt_embeds=torch.zeros_like(embeds), negative_prompt_embeds_mask=mask,
                    width=64, height=64, num_inference_steps=2, sigmas=beta_sigmas(2, 1), guidance_scale=cfg-1,
                    generator=torch.Generator().manual_seed(42), output_type="np")
                assert result.images.shape == (1, 64, 64, 3)
                assert torch.isfinite(torch.from_numpy(result.images)).all()
                assert pipe.do_classifier_free_guidance == (cfg != 1)
                if retain_device:
                    assert tuple(pipe._all_hooks) == hooks
                    assert len(resets) == iteration + 1
                    assert next(vae.parameters()).device.type == retain_device
                    assert next(transformer.parameters()).device.type == retain_device
                    if iteration:
                        torch.testing.assert_close(torch.from_numpy(result.images), first)
                    first = torch.from_numpy(result.images).clone()
    finally:
        pipe.remove_all_hooks()
        pipe.to("cpu", silence_dtype_warnings=True)
