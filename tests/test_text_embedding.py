from __future__ import annotations

import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from pulid_app.config import TextEmbeddingConfig
from pulid_app.models.text_embedding import (
    TextEmbeddingService,
    _cuda_dll_candidates,
    _nvidia_driver_cuda_dll_directories,
    _prepend_dll_directories_to_path,
    load_llama_cpp_embedding_model,
)


def test_cuda_dll_candidates_exclude_global_torch_and_cuda_toolkit(tmp_path: Path, monkeypatch) -> None:
    prefix = tmp_path / "venv"
    prefix_torch_lib = prefix / "Lib" / "site-packages" / "torch" / "lib"
    prefix_torch_lib.mkdir(parents=True)
    discovered_torch = tmp_path / "discovered" / "torch"
    (discovered_torch / "lib").mkdir(parents=True)
    cuda_root = tmp_path / "cuda"
    (cuda_root / "bin").mkdir(parents=True)

    monkeypatch.setenv("CUDA_PATH", str(cuda_root))
    monkeypatch.setenv("CUDA_PATH_V13_0", str(cuda_root))
    monkeypatch.setenv("PATH", str(discovered_torch / "lib"))
    candidates = _cuda_dll_candidates(prefix=prefix)

    assert candidates == (prefix_torch_lib.resolve(),)


def test_cuda_dll_candidates_include_nvidia_driver_store(tmp_path: Path) -> None:
    windows_root = tmp_path / "Windows"
    repository = windows_root / "System32" / "DriverStore" / "FileRepository"
    driver_dir = repository / "nv_dispi.inf_amd64_current"
    driver_dir.mkdir(parents=True)
    (driver_dir / "nvcudart_hybrid64.dll").touch()

    candidates = _cuda_dll_candidates(
        prefix=tmp_path / "venv",
        windows_root=windows_root,
    )

    assert candidates == (driver_dir.resolve(),)


def test_nvidia_driver_cuda_dll_directories_prefer_newest(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "System32" / "DriverStore" / "FileRepository"
    older = repository / "nv_old.inf_amd64_1" / "nvcudart_hybrid64.dll"
    newer = repository / "nv_new.inf_amd64_2" / "nvcudart_hybrid64.dll"
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.touch()
    newer.touch()
    os.utime(older, ns=(1, 1))
    os.utime(newer, ns=(2, 2))

    directories = _nvidia_driver_cuda_dll_directories(tmp_path)

    assert directories == (newer.parent,)


def test_cuda_dll_directories_are_prepended_to_path_without_duplicates() -> None:
    torch_lib = Path(r"D:\PuLID\.venv\Lib\site-packages\torch\lib")
    driver_store = Path(r"C:\Windows\System32\DriverStore\NVIDIA")

    result = _prepend_dll_directories_to_path(
        rf"C:\Windows\System32;{torch_lib}",
        (torch_lib, driver_store),
        separator=";",
    )

    assert result == (
        rf"{torch_lib};{driver_store};C:\Windows\System32"
    )


def test_llama_cpp_factory_forces_cpu_and_local_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "bge-m3-Q8_0.gguf"
    checkpoint.touch()
    captured = {}
    sentinel = object()

    def fake_llama(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=fake_llama, llama_supports_gpu_offload=lambda: True, llama_print_system_info=lambda: b"MTL CUDA"))
    config = TextEmbeddingConfig(
        checkpoint=checkpoint,
        context_size=4096,
        batch_size=4096,
        threads=2,
    )

    result = load_llama_cpp_embedding_model(config)

    assert result is sentinel
    assert captured == {
        "model_path": str(checkpoint),
        "embedding": True,
        "n_gpu_layers": 0,
        "n_ctx": 4096,
        "n_batch": 4096,
        "n_ubatch": 4096,
        "n_threads": 2,
        "n_threads_batch": 2,
        "offload_kqv": False,
        "op_offload": False,
        "flash_attn": False,
        "use_mmap": True,
        "verbose": False,
    }


def test_llama_cpp_factory_uses_runtime_thread_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "bge-m3-Q8_0.gguf"
    checkpoint.touch()
    captured = {}

    def fake_llama(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=fake_llama, llama_supports_gpu_offload=lambda: True, llama_print_system_info=lambda: b"MTL CUDA"))

    load_llama_cpp_embedding_model(TextEmbeddingConfig(checkpoint=checkpoint))

    assert "n_threads" not in captured
    assert "n_threads_batch" not in captured


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_llama_cpp_factory_offloads_all_layers_without_reducing_context(
    tmp_path: Path,
    monkeypatch,
    device: str,
) -> None:
    checkpoint = tmp_path / "bge-m3-Q8_0.gguf"
    checkpoint.touch()
    captured = {}

    def fake_llama(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=fake_llama, llama_supports_gpu_offload=lambda: True, llama_print_system_info=lambda: b"MTL CUDA"))
    config = TextEmbeddingConfig(checkpoint=checkpoint)

    load_llama_cpp_embedding_model(config, device=device)

    assert captured["n_gpu_layers"] == -1
    assert captured["offload_kqv"] is True
    assert captured["op_offload"] is True
    assert captured["flash_attn"] is True
    assert captured["n_ctx"] == 8192
    assert captured["n_batch"] == 8192
    assert captured["n_ubatch"] == 8192
    assert captured["verbose"] is False


def test_embedding_service_rejects_token_overflow_without_truncating(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "bge-m3-Q8_0.gguf"
    checkpoint.touch()

    class TokenOverflowEngine:
        called = False

        def tokenize(self, _content):
            return list(range(129))

        def create_embedding(self, _inputs):
            self.called = True
            raise AssertionError("Le calcul ne doit pas démarrer.")

    engine = TokenOverflowEngine()
    service = TextEmbeddingService(
        TextEmbeddingConfig(checkpoint=checkpoint, context_size=128),
        model_factory=lambda _config, **_kwargs: engine,
    )

    with pytest.raises(ValueError, match="fenêtre configurée est de 128 jetons"):
        service.create_embedding(
            model="text-embedding-bge-m3",
            input_value="texte trop long",
        )

    assert engine.called is False


@pytest.mark.parametrize('device', ['mps', 'cuda'])
def test_gpu_embedding_rejects_cpu_wheel_without_fallback(tmp_path, monkeypatch, device):
    from pulid_app.exceptions import ModelLoadError
    checkpoint = tmp_path / 'bge.gguf'
    checkpoint.touch()

    def forbidden_load(**kwargs):
        raise AssertionError('A CPU model must never be loaded for a GPU request')

    monkeypatch.setitem(sys.modules, 'llama_cpp', SimpleNamespace(
        Llama=forbidden_load, llama_supports_gpu_offload=lambda: False,
        llama_print_system_info=lambda: b'CPU : NEON = 1',
    ))
    with pytest.raises(ModelLoadError, match='Aucun repli CPU'):
        load_llama_cpp_embedding_model(TextEmbeddingConfig(checkpoint=checkpoint), device=device)
