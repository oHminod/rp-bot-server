from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
from PIL import Image
import pytest

from pulid_app import __version__
from pulid_app.api_contract import API_CONTRACT_VERSION
from pulid_app.exceptions import PromptTooLongError, UnsupportedDeviceError
from pulid_app.server import (
    build_parser,
    create_app,
    generated_filename,
    resolve_generation_seed,
    resolve_network_settings,
)


def _write_config(tmp_path: Path) -> tuple[Path, Path]:
    models = tmp_path / "models"
    checkpoints = models / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "realvisxl.safetensors").touch()
    (checkpoints / "reaxl_v30.safetensors").touch()
    text_embedding = models / "text_embedding" / "bge-m3-Q8_0.gguf"
    text_embedding.parent.mkdir()
    text_embedding.touch()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
models_root: {models}
sdxl:
  checkpoint: checkpoints/realvisxl.safetensors
  config_dir: sdxl/config
pulid:
  checkpoint: pulid_v1.1.safetensors
insightface:
  model_root: .
  model_name: antelopev2
text_embedding:
  checkpoint: text_embedding/bge-m3-Q8_0.gguf
  model_id: text-embedding-bge-m3
  dimensions: 2
  context_size: 8192
  batch_size: 8192
  threads: 0
outputs_dir: {tmp_path / 'outputs'}
identity_cache_dir: {tmp_path / 'cache' / 'identity'}
device:
  preferred: cpu
  dtype: float32
  offload_strategy: none
""",
        encoding="utf-8",
    )
    return config, models


def _image_bytes(format_: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (24, 24), "white").save(output, format=format_)
    return output.getvalue()


class FakeMemoryGenerator:
    instances: list["FakeMemoryGenerator"] = []

    def __init__(self, config, **kwargs) -> None:
        self.config = config
        self.constructor_kwargs = kwargs
        self.encode_kwargs = None
        self.generate_kwargs = None
        self.closed = False
        self.partial_offload_calls = 0
        self.restore_calls = 0
        self.full_offload_calls = 0
        self.partially_offloaded = False
        self.__class__.instances.append(self)

    def encode_identity_memory(self, image, **kwargs):
        self.encode_kwargs = {"image": image, **kwargs}
        return object()

    def generate_in_memory(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(
            image=Image.new("RGB", (32, 32), "navy"),
            metadata={},
        )

    def close(self) -> None:
        self.closed = True

    def partial_offload_sdxl_for_embedding(self) -> bool:
        if self.partially_offloaded:
            return False
        self.partial_offload_calls += 1
        self.partially_offloaded = True
        return True

    def restore_sdxl_after_embedding(self) -> bool:
        if not self.partially_offloaded:
            return False
        self.restore_calls += 1
        self.partially_offloaded = False
        return True

    def full_offload_sdxl_for_embedding(self) -> bool:
        self.full_offload_calls += 1
        return True


class FakeEmbeddingModel:
    instances: list["FakeEmbeddingModel"] = []

    def __init__(self, config, *, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def create_embedding(self, inputs):
        self.calls.append(inputs)
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": [3.0, 4.0],
                }
                for index, _text in enumerate(inputs)
            ],
            "model": str(self.config.checkpoint),
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
        }

    def close(self) -> None:
        self.closed = True


def _app(
    tmp_path: Path,
    *,
    device: str | None = None,
    embedding_memory_mode: str = "cpu",
    cors_origins: tuple[str, ...] = (),
):
    config, _models = _write_config(tmp_path)
    FakeMemoryGenerator.instances.clear()
    FakeEmbeddingModel.instances.clear()
    return create_app(
        config,
        device=device,
        embedding_memory_mode=embedding_memory_mode,
        generator_factory=FakeMemoryGenerator,
        embedding_model_factory=FakeEmbeddingModel,
        now_factory=lambda: datetime(2026, 8, 13, 20, 9, 47, 123456, tzinfo=timezone.utc),
        random_seed=lambda: 987654321,
        cors_origins=cors_origins,
    )


def _request(app, method: str, path: str, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def _generate_request(app, model: str = "realvisxl"):
    return _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": model,
            "seed": "42",
        },
    )


def test_discovery_endpoints_are_lightweight_and_versioned(tmp_path: Path) -> None:
    app = _app(tmp_path)
    for model_file in tmp_path.glob("models/**/*"):
        if model_file.is_file():
            model_file.unlink()

    health = _request(app, "GET", "/health")
    version = _request(app, "GET", "/version")
    capabilities = _request(app, "GET", "/capabilities")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "version": __version__,
        "api_contract_version": API_CONTRACT_VERSION,
    }
    assert version.status_code == 200
    assert version.json() == {
        "component": "pulid",
        "version": __version__,
        "api_contract_version": API_CONTRACT_VERSION,
    }
    assert capabilities.status_code == 200
    assert capabilities.json() == {
        "component": "pulid",
        "version": __version__,
        "api_contract_version": API_CONTRACT_VERSION,
        "capabilities": {
            "image_generation": {
                "enabled": True,
                "catalog_endpoint": "/models",
                "generation_endpoint": "/generate",
            },
            "text_embeddings": {
                "enabled": True,
                "models_endpoint": "/v1/models",
                "embeddings_endpoint": "/v1/embeddings",
                "model": "text-embedding-bge-m3",
                "dimensions": 2,
            },
        },
    }
    assert FakeMemoryGenerator.instances == []
    assert FakeEmbeddingModel.instances == []


def test_models_endpoint_lists_sampling_methods_and_sigmas_separately(
    tmp_path: Path,
) -> None:
    response = _request(_app(tmp_path), "GET", "/models")

    assert response.status_code == 200
    assert response.json() == {
        "models": [
            {
                "name": "realvisxl",
                "filename": "realvisxl.safetensors",
                "default": True,
            },
            {
                "name": "reaxl_v30",
                "filename": "reaxl_v30.safetensors",
                "default": False,
            },
        ],
        "sampling_methods": [
            {
                "name": "default",
                "label": "Scheduler du checkpoint",
                "default": True,
                "supported_sigma_schedules": ["normal"],
            },
            {
                "name": "dpmpp_2m",
                "label": "DPM++ 2M",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "dpmpp_2m_sde",
                "label": "DPM++ 2M SDE",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "dpmpp_3m_sde",
                "label": "DPM++ 3M SDE",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "euler",
                "label": "Euler",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "euler_ancestral",
                "label": "Euler ancestral",
                "default": False,
                "supported_sigma_schedules": ["normal"],
            },
            {
                "name": "heun",
                "label": "Heun",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "lms",
                "label": "LMS",
                "default": False,
                "supported_sigma_schedules": [
                    "normal",
                    "karras",
                    "exponential",
                    "beta",
                ],
            },
            {
                "name": "ddim",
                "label": "DDIM",
                "default": False,
                "supported_sigma_schedules": ["normal"],
            },
        ],
        "sigma_schedules": [
            {
                "name": "normal",
                "label": "Normal / natif",
                "default": True,
                "supported_sampling_methods": [
                    "default",
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "euler_ancestral",
                    "heun",
                    "lms",
                    "ddim",
                ],
            },
            {
                "name": "karras",
                "label": "Karras",
                "default": False,
                "supported_sampling_methods": [
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "heun",
                    "lms",
                ],
            },
            {
                "name": "exponential",
                "label": "Exponentiel",
                "default": False,
                "supported_sampling_methods": [
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "heun",
                    "lms",
                ],
            },
            {
                "name": "beta",
                "label": "Beta",
                "default": False,
                "supported_sampling_methods": [
                    "dpmpp_2m",
                    "dpmpp_2m_sde",
                    "dpmpp_3m_sde",
                    "euler",
                    "heun",
                    "lms",
                ],
            },
        ],
    }


def test_openai_models_endpoint_lists_configured_embedding(tmp_path: Path) -> None:
    response = _request(_app(tmp_path), "GET", "/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": "text-embedding-bge-m3",
                "object": "model",
                "created": 0,
                "owned_by": "pulid-local",
            }
        ],
    }
    assert FakeEmbeddingModel.instances == []


def test_openai_embeddings_endpoint_is_lazy_reused_and_normalized(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)

    first = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "text-embedding-bge-m3", "input": "bonjour"},
    )
    second = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={
            "model": "text-embedding-bge-m3",
            "input": ["bonjour", "au revoir"],
        },
    )

    assert first.status_code == 200
    assert first.json() == {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.6, 0.8],
            }
        ],
        "model": "text-embedding-bge-m3",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }
    assert second.status_code == 200
    assert len(second.json()["data"]) == 2
    assert len(FakeEmbeddingModel.instances) == 1
    engine = FakeEmbeddingModel.instances[0]
    assert engine.config.checkpoint == (
        tmp_path / "models" / "text_embedding" / "bge-m3-Q8_0.gguf"
    )
    assert engine.calls == [["bonjour"], ["bonjour", "au revoir"]]


def test_openai_embeddings_rejects_unknown_model_without_loading_gguf(
    tmp_path: Path,
) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/v1/embeddings",
        json={"model": "unknown", "input": "bonjour"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "ValueError"
    assert FakeEmbeddingModel.instances == []


def test_openai_embeddings_reports_missing_gguf_as_unavailable(tmp_path: Path) -> None:
    app = _app(tmp_path)
    (tmp_path / "models" / "text_embedding" / "bge-m3-Q8_0.gguf").unlink()

    response = _request(app, "GET", "/v1/models")

    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "ModelNotFoundError"
    assert "bge-m3-Q8_0.gguf" in response.json()["detail"]["message"]
    assert FakeEmbeddingModel.instances == []


def test_embedding_model_is_closed_on_app_shutdown(tmp_path: Path) -> None:
    app = _app(tmp_path)

    async def run_with_lifespan() -> FakeEmbeddingModel:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/v1/embeddings",
                    json={
                        "model": "text-embedding-bge-m3",
                        "input": "bonjour",
                    },
                )
            assert response.status_code == 200
            engine = FakeEmbeddingModel.instances[0]
            assert engine.closed is False
            return engine

    engine = asyncio.run(run_with_lifespan())

    assert engine.closed is True


def test_cpu_embedding_can_run_during_sdxl_generation(tmp_path: Path) -> None:
    generation_started = threading.Event()
    embedding_started = threading.Event()

    class ConcurrentGenerator(FakeMemoryGenerator):
        def generate_in_memory(self, **kwargs):
            generation_started.set()
            if not embedding_started.wait(timeout=2):
                raise RuntimeError("L'embedding n'a pas démarré en parallèle.")
            return super().generate_in_memory(**kwargs)

    class ConcurrentEmbeddingModel(FakeEmbeddingModel):
        def create_embedding(self, inputs):
            if not generation_started.wait(timeout=2):
                raise RuntimeError("La génération SDXL n'a pas démarré.")
            embedding_started.set()
            return super().create_embedding(inputs)

    config, _models = _write_config(tmp_path)
    ConcurrentGenerator.instances.clear()
    ConcurrentEmbeddingModel.instances.clear()
    app = create_app(
        config,
        embedding_memory_mode="cpu",
        generator_factory=ConcurrentGenerator,
        embedding_model_factory=ConcurrentEmbeddingModel,
    )

    async def send_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            generation = asyncio.create_task(
                client.post(
                    "/generate",
                    files={
                        "reference": ("noemie.png", _image_bytes(), "image/png")
                    },
                    data={
                        "character": "noemie",
                        "prompt": "portrait",
                        "model": "realvisxl",
                    },
                )
            )
            embedding = asyncio.create_task(
                client.post(
                    "/v1/embeddings",
                    json={
                        "model": "text-embedding-bge-m3",
                        "input": "bonjour",
                    },
                )
            )
            return await asyncio.gather(generation, embedding)

    generation_response, embedding_response = asyncio.run(send_concurrently())

    assert generation_response.status_code == 200
    assert embedding_response.status_code == 200


def test_embedding_memory_cli_modes_are_exclusive() -> None:
    parser = build_parser()

    defaults = parser.parse_args([])
    assert defaults.embedding_memory_mode == "concurrent"
    assert defaults.host == "127.0.0.1"
    assert defaults.network is False
    assert defaults.cors_origin == []
    network = parser.parse_args(["--network"])
    assert network.network is True
    assert resolve_network_settings(
        network.host,
        network.cors_origin,
        network=network.network,
    ) == ("0.0.0.0", ("*",))
    assert resolve_network_settings(
        "192.168.1.20",
        ("http://localhost:8800", "*"),
        network=True,
    ) == ("0.0.0.0", ("http://localhost:8800", "*"))
    assert parser.parse_args(["--partial"]).embedding_memory_mode == "partial"
    assert parser.parse_args(["--full"]).embedding_memory_mode == "full"
    assert (
        parser.parse_args(["--serialized-cuda"]).embedding_memory_mode
        == "serialized"
    )
    assert parser.parse_args(["--CPU"]).embedding_memory_mode == "cpu"
    with pytest.raises(SystemExit):
        parser.parse_args(["--partial", "--full"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--serialized-cuda", "--CPU"])


def test_gpu_embedding_modes_require_an_accelerator(tmp_path: Path) -> None:
    config, _models = _write_config(tmp_path)

    with pytest.raises(UnsupportedDeviceError, match="Utilisez --CPU"):
        create_app(config, device="cpu", embedding_memory_mode="concurrent")


def test_default_mode_remains_serialized_on_mps(tmp_path: Path) -> None:
    app = _app(tmp_path, device="mps", embedding_memory_mode="concurrent")

    assert app.state.concurrent_cuda is False


def test_cpu_mode_forces_embedding_to_cpu_without_sdxl_offload(tmp_path: Path) -> None:
    app = _app(tmp_path, device="cuda", embedding_memory_mode="cpu")
    assert _generate_request(app).status_code == 200

    embedding = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "text-embedding-bge-m3", "input": "bonjour"},
    )

    assert embedding.status_code == 200
    assert FakeEmbeddingModel.instances[0].device == "cpu"
    generator = FakeMemoryGenerator.instances[0]
    assert generator.partial_offload_calls == 0
    assert generator.full_offload_calls == 0


def test_default_gpu_mode_keeps_sdxl_and_bge_loaded_together(tmp_path: Path) -> None:
    app = _app(tmp_path, device="cuda", embedding_memory_mode="concurrent")

    generation = _generate_request(app)
    embedding = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "text-embedding-bge-m3", "input": "bonjour"},
    )

    assert generation.status_code == 200
    assert embedding.status_code == 200
    generator = FakeMemoryGenerator.instances[0]
    engine = FakeEmbeddingModel.instances[0]
    assert engine.device == "cuda"
    assert engine.closed is False
    assert generator.closed is False
    assert generator.partial_offload_calls == 0
    assert generator.full_offload_calls == 0


def test_partial_mode_parks_sdxl_until_next_generation(tmp_path: Path) -> None:
    app = _app(tmp_path, device="cuda", embedding_memory_mode="partial")
    assert _generate_request(app).status_code == 200
    generator = FakeMemoryGenerator.instances[0]

    first_embedding = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "text-embedding-bge-m3", "input": "bonjour"},
    )
    second_embedding = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "text-embedding-bge-m3", "input": "encore"},
    )
    engine = FakeEmbeddingModel.instances[0]

    assert first_embedding.status_code == 200
    assert second_embedding.status_code == 200
    assert generator.partial_offload_calls == 1
    assert generator.restore_calls == 0
    assert engine.closed is False

    assert _generate_request(app).status_code == 200
    assert generator.restore_calls == 1
    assert engine.closed is True
    assert generator.closed is False


def test_full_mode_unloads_only_sdxl_until_next_generation(tmp_path: Path) -> None:
    app = _app(tmp_path, device="cuda", embedding_memory_mode="full")
    assert _generate_request(app).status_code == 200
    generator = FakeMemoryGenerator.instances[0]

    embedding = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "text-embedding-bge-m3", "input": "bonjour"},
    )
    engine = FakeEmbeddingModel.instances[0]

    assert embedding.status_code == 200
    assert generator.full_offload_calls == 1
    assert generator.restore_calls == 0
    assert generator.closed is False

    assert _generate_request(app).status_code == 200
    assert engine.closed is True
    assert generator.restore_calls == 0
    assert generator.closed is False


@pytest.mark.parametrize(
    ("embedding_memory_mode", "expected_peak"),
    (("serialized", 1), ("concurrent", 2)),
)
def test_gpu_embedding_concurrency_policy(
    tmp_path: Path,
    embedding_memory_mode: str,
    expected_peak: int,
) -> None:
    state_lock = threading.Lock()
    active = 0
    peak = 0

    def occupy_accelerator() -> None:
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1

    class SerializedGenerator(FakeMemoryGenerator):
        def generate_in_memory(self, **kwargs):
            occupy_accelerator()
            return super().generate_in_memory(**kwargs)

    class SerializedEmbeddingModel(FakeEmbeddingModel):
        def create_embedding(self, inputs):
            occupy_accelerator()
            return super().create_embedding(inputs)

    config, _models = _write_config(tmp_path)
    SerializedGenerator.instances.clear()
    SerializedEmbeddingModel.instances.clear()
    app = create_app(
        config,
        device="cuda",
        embedding_memory_mode=embedding_memory_mode,
        generator_factory=SerializedGenerator,
        embedding_model_factory=SerializedEmbeddingModel,
    )

    async def send_concurrently():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            generation = asyncio.create_task(
                client.post(
                    "/generate",
                    files={"reference": ("noemie.png", _image_bytes(), "image/png")},
                    data={
                        "character": "noemie",
                        "prompt": "portrait",
                        "model": "realvisxl",
                    },
                )
            )
            embedding = asyncio.create_task(
                client.post(
                    "/v1/embeddings",
                    json={"model": "text-embedding-bge-m3", "input": "bonjour"},
                )
            )
            return await asyncio.gather(generation, embedding)

    generation_response, embedding_response = asyncio.run(send_concurrently())

    assert generation_response.status_code == 200
    assert embedding_response.status_code == 200
    assert peak == expected_peak


def test_generate_returns_png_headers_and_enables_identity_cache(tmp_path: Path) -> None:
    app = _app(tmp_path)
    files_before = {path for path in tmp_path.rglob("*") if path.is_file()}

    response = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.webp", _image_bytes("WEBP"), "image/webp")},
        data={
            "character": "Noémie",
            "prompt": "cinematic portrait",
            "negative_prompt": "bad anatomy, watermark",
            "clip_skip_2": "true",
            "model": "reaxl_v30",
            "cfg": "4.5",
            "steps": "8",
            "strength": "1.25",
            "method": "dpmpp_2m_sde",
            "sigmas": "karras",
            "seed": "0",
            "width": "832",
            "height": "1216",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-generation-seed"] == "987654321"
    assert response.headers["x-sdxl-model"] == "reaxl_v30"
    assert response.headers["x-sampling-method"] == "dpmpp_2m_sde"
    assert response.headers["x-sigma-schedule"] == "karras"
    assert response.headers["content-disposition"] == (
        'attachment; filename="noemie_20260813T200947_123456Z.png"'
    )
    with Image.open(BytesIO(response.content)) as generated:
        assert generated.size == (32, 32)

    instance = FakeMemoryGenerator.instances[0]
    assert instance.config.sdxl.checkpoint == (
        tmp_path / "models" / "checkpoints" / "reaxl_v30.safetensors"
    )
    assert instance.encode_kwargs["identity_id"] == "Noémie"
    assert instance.encode_kwargs["source_name"] == "<http-upload>"
    assert instance.encode_kwargs["use_cache"] is True
    assert instance.generate_kwargs["seed"] == 987654321
    assert instance.generate_kwargs["negative_prompt"] == "bad anatomy, watermark"
    assert instance.generate_kwargs["clip_skip_2"] is True
    assert instance.generate_kwargs["steps"] == 8
    assert instance.generate_kwargs["guidance_scale"] == 4.5
    assert instance.generate_kwargs["identity_strength"] == 1.25
    assert instance.generate_kwargs["sampling_method"] == "dpmpp_2m_sde"
    assert instance.generate_kwargs["sigma_schedule"] == "karras"
    assert instance.generate_kwargs["width"] == 832
    assert instance.generate_kwargs["height"] == 1216
    assert instance.closed is True
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == files_before
    assert not (tmp_path / "outputs").exists()


def test_generate_accepts_default_method_and_explicit_seed(tmp_path: Path) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "seed": "42",
        },
    )

    assert response.status_code == 200
    assert response.headers["x-generation-seed"] == "42"
    assert response.headers["x-sampling-method"] == "default"
    assert response.headers["x-sigma-schedule"] == "normal"
    assert FakeMemoryGenerator.instances[0].generate_kwargs["sampling_method"] is None
    assert FakeMemoryGenerator.instances[0].generate_kwargs["sigma_schedule"] is None
    assert FakeMemoryGenerator.instances[0].generate_kwargs["identity_strength"] == 0.8
    assert FakeMemoryGenerator.instances[0].generate_kwargs["width"] == 1024
    assert FakeMemoryGenerator.instances[0].generate_kwargs["height"] == 1024
    assert (
        FakeMemoryGenerator.instances[0].generate_kwargs["negative_prompt"]
        == "flaws in the eyes, flaws in the face, low quality, worst quality, "
        "artifacts, text, watermark, deformed, mutated, disfigured, blurry"
    )
    assert FakeMemoryGenerator.instances[0].generate_kwargs["clip_skip_2"] is False


def test_cuda_reuses_generator_for_same_checkpoint_until_service_close(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path, device="cuda")

    first = _generate_request(app)
    second = _generate_request(app)

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(FakeMemoryGenerator.instances) == 1
    generator = FakeMemoryGenerator.instances[0]
    assert generator.closed is False

    app.state.generation_service.close()
    app.state.generation_service.close()

    assert generator.closed is True


def test_cuda_closes_previous_generator_when_checkpoint_changes(tmp_path: Path) -> None:
    app = _app(tmp_path, device="cuda")

    first = _generate_request(app, "realvisxl")
    second = _generate_request(app, "reaxl_v30")

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(FakeMemoryGenerator.instances) == 2
    previous, current = FakeMemoryGenerator.instances
    assert previous.closed is True
    assert current.closed is False
    assert current.config.sdxl.checkpoint.name == "reaxl_v30.safetensors"

    app.state.generation_service.close()

    assert current.closed is True


def test_app_shutdown_closes_retained_cuda_generator(tmp_path: Path) -> None:
    app = _app(tmp_path, device="cuda")

    async def run_with_lifespan() -> FakeMemoryGenerator:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/generate",
                    files={
                        "reference": ("noemie.png", _image_bytes(), "image/png")
                    },
                    data={
                        "character": "noemie",
                        "prompt": "portrait",
                        "model": "realvisxl",
                        "seed": "42",
                    },
                )
            assert response.status_code == 200
            generator = FakeMemoryGenerator.instances[0]
            assert generator.closed is False
            return generator

    generator = asyncio.run(run_with_lifespan())

    assert generator.closed is True


def test_mps_still_closes_generator_after_each_request(tmp_path: Path) -> None:
    app = _app(tmp_path, device="mps")

    first = _generate_request(app)
    second = _generate_request(app)

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(FakeMemoryGenerator.instances) == 2
    assert all(generator.closed for generator in FakeMemoryGenerator.instances)


@pytest.mark.parametrize("strength", ["-0.1", "nan", "inf"])
def test_generate_rejects_invalid_strength_before_loading_generator(
    tmp_path: Path,
    strength: str,
) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "strength": strength,
        },
    )

    assert response.status_code == 422
    assert FakeMemoryGenerator.instances == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", "63"),
        ("width", "1025"),
        ("height", "2056"),
    ],
)
def test_generate_rejects_invalid_optional_dimensions_before_loading_generator(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            field: value,
        },
    )

    assert response.status_code == 422
    assert FakeMemoryGenerator.instances == []


def test_generate_rejects_unknown_model_without_loading_generator(tmp_path: Path) -> None:
    response = _request(
        _app(tmp_path),
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "absent",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "ModelNotFoundError"
    assert FakeMemoryGenerator.instances == []


def test_generate_reports_long_prompt_as_a_422_client_error(tmp_path: Path) -> None:
    class LongPromptRejectingGenerator(FakeMemoryGenerator):
        def generate_in_memory(self, **kwargs):
            self.generate_kwargs = kwargs
            raise PromptTooLongError(
                prompt_kind="prompt positif",
                token_count=256,
                max_tokens=255,
                encoder_index=1,
            )

    config, _models = _write_config(tmp_path)
    app = create_app(
        config,
        embedding_memory_mode="cpu",
        generator_factory=LongPromptRejectingGenerator,
    )

    response = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "very long prompt",
            "model": "realvisxl",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "PromptTooLongError"
    assert "256 jetons utiles" in response.json()["detail"]["message"]
    assert LongPromptRejectingGenerator.instances[-1].closed is True


def test_generate_rejects_invalid_image_sampling_method_and_sigmas(
    tmp_path: Path,
) -> None:
    app = _app(tmp_path)
    invalid_image = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("bad.png", b"not-an-image", "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
        },
    )
    invalid_method = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "method": "unknown",
        },
    )
    invalid_sigmas = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "method": "euler",
            "sigmas": "unknown",
        },
    )
    incompatible_sigmas = _request(
        app,
        "POST",
        "/generate",
        files={"reference": ("noemie.png", _image_bytes(), "image/png")},
        data={
            "character": "noemie",
            "prompt": "portrait",
            "model": "realvisxl",
            "method": "euler_ancestral",
            "sigmas": "karras",
        },
    )

    assert invalid_image.status_code == 422
    assert invalid_method.status_code == 422
    assert invalid_sigmas.status_code == 422
    assert incompatible_sigmas.status_code == 422
    assert "Méthode de sampling inconnue" in invalid_method.json()["detail"]["message"]
    assert "Courbe de sigmas inconnue" in invalid_sigmas.json()["detail"]["message"]
    assert "incompatible" in incompatible_sigmas.json()["detail"]["message"]
    assert FakeMemoryGenerator.instances == []


def test_seed_and_filename_helpers_are_deterministic() -> None:
    created = datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=timezone.utc)

    assert resolve_generation_seed(0, lambda: 123) == 123
    assert resolve_generation_seed(-1, lambda: 456) == 456
    assert resolve_generation_seed(42) == 42
    assert generated_filename("Noémie Test", created) == (
        "noemie-test_20260102T030405_000006Z.png"
    )


def test_configured_cors_origin_can_call_frontend_endpoints(tmp_path: Path) -> None:
    app = _app(tmp_path, cors_origins=("http://localhost:3000",))

    response = _request(
        app,
        "OPTIONS",
        "/generate",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_server_starts_without_sdxl_and_ignores_broken_uv_tree(tmp_path):
    from fastapi.testclient import TestClient
    config, models = _write_config(tmp_path)
    for checkpoint in (models / 'checkpoints').iterdir():
        checkpoint.unlink()
    runtime = models / 'other' / 'uv-python-windows'
    runtime.mkdir(parents=True)
    try:
        (runtime / 'cpython-3.11-windows-x86_64-none').symlink_to(
            runtime / 'missing', target_is_directory=True,
        )
    except OSError:
        pass  # The independent traversal tests cover denied symlink creation.

    def forbidden_load(*args, **kwargs):
        raise AssertionError('Startup and catalog must not load any model')

    app = create_app(config, device='cpu', embedding_memory_mode='cpu',
                     generator_factory=forbidden_load, embedding_model_factory=forbidden_load)
    with TestClient(app) as client:
        response = client.get('/health')
        assert response.status_code == 200
        response = client.get('/models')
        assert response.status_code == 200
        assert response.json()['models'] == []
