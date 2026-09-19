"""Régressions des chemins CUDA, avec calculs synthétiques et sans poids externes."""
from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pulid_app.exceptions import ModelLoadError
from pulid_app.paths import configure_external_model_caches


@pytest.fixture
def runtime(tmp_path):
    configure_external_model_caches(tmp_path)
    import torch
    return torch


def fake_kitchen(monkeypatch, backend, registry=None):
    kitchen = ModuleType("comfy_kitchen")
    kitchen.list_backends = lambda: {"cuda": backend}
    registry_module = ModuleType("comfy_kitchen.registry")
    registry_module.registry = registry
    monkeypatch.setitem(sys.modules, "comfy_kitchen", kitchen)
    monkeypatch.setitem(sys.modules, "comfy_kitchen.registry", registry_module)


@pytest.mark.parametrize("backend", [
    {"available": False, "unavailable_reason": "CUDA DLL absente"},
    {"available": True, "disabled": True},
])
def test_missing_cuda_backend_fails_with_install_action(monkeypatch, backend):
    from pulid_app.models.quantized_cuda import CudaQuantizedKernels
    fake_kitchen(monkeypatch, backend)
    with pytest.raises(ModelLoadError, match="install_windows.bat"):
        CudaQuantizedKernels()


@pytest.mark.parametrize("format", ["nvfp4", "float8_e4m3fn"])
def test_kernel_adapter_preserves_packed_bits_and_forces_cuda(runtime, monkeypatch, format):
    torch = runtime
    from pulid_app.models.quantized import QuantizedSpec
    from pulid_app.models.quantized_cuda import CudaQuantizedKernels

    packed = torch.zeros((16, 16), dtype=torch.uint8)
    scales = (torch.ones((128, 4)).to(torch.float8_e4m3fn) if format == "nvfp4"
              else torch.tensor([1e-8])).view(torch.uint8)
    global_scale = torch.tensor([1e-8]).view(torch.uint8)
    calls = []
    def resolve(function, *, backend, kwargs):
        calls.append((function, backend))
        def kernel(**arguments):
            assert arguments["output_type"] == torch.float16
            if format == "nvfp4":
                assert arguments["qx"] is packed
                assert arguments["block_scales"].data_ptr() == scales.data_ptr()
                assert arguments["block_scales"].dtype == torch.float8_e4m3fn
                torch.testing.assert_close(arguments["per_tensor_scale"], global_scale.view(torch.float32))
                assert arguments["hi_first"] is True
            else:
                assert arguments["x"].data_ptr() == packed.data_ptr()
                assert arguments["x"].dtype == torch.float8_e4m3fn
                torch.testing.assert_close(arguments["scale"], scales.view(torch.float32))
            return "decoded"
        return kernel
    fake_kitchen(monkeypatch, {"available": True}, SimpleNamespace(get_implementation=resolve))
    kernels = CudaQuantizedKernels()
    spec = QuantizedSpec(format, (16, 32))
    assert kernels.dequantize(packed, scales, global_scale, spec, torch.float16) == "decoded"
    expected = "dequantize_nvfp4" if format == "nvfp4" else "dequantize_per_tensor_fp8"
    assert calls == [(expected, "cuda")]
    def unavailable(*args, **kwargs):
        raise RuntimeError("kernel constraints failed")
    monkeypatch.setattr(kernels.registry, "get_implementation", unavailable)
    with pytest.raises(RuntimeError, match="kernel constraints failed"):
        kernels.dequantize(packed, scales, global_scale, spec, torch.float16)


def test_cuda_kernels_shared_only_when_quantized_models_need_them(runtime, tmp_path, monkeypatch):
    from test_quantized_krea2 import load_fp8_layer
    from pulid_app.models import quantized_cuda
    layer, _ = load_fp8_layer(runtime, tmp_path)
    calls = []
    kernels = object()
    def create():
        calls.append(True)
        return kernels
    monkeypatch.setattr(quantized_cuda, "CudaQuantizedKernels", create)
    for device in ("cpu", "mps"):
        quantized_cuda.configure_cuda_kernels((layer,), device)
    quantized_cuda.configure_cuda_kernels((runtime.nn.Linear(1, 1),), "cuda")
    assert not calls and layer.cuda_kernels is None
    other, _ = load_fp8_layer(runtime, tmp_path)
    quantized_cuda.configure_cuda_kernels((layer, other), "cuda:0")
    assert len(calls) == 1
    assert layer.cuda_kernels is other.cuda_kernels is kernels


@pytest.mark.parametrize("mask_kind", ["none", "bool", "float"])
def test_expanded_gqa_attention_matches_diffusers_with_rotary(runtime, monkeypatch, mask_kind):
    torch = runtime
    from diffusers.models.transformers.transformer_krea2 import Krea2Attention
    from pulid_app.models import krea2_attention

    torch.manual_seed(4)
    attention = Krea2Attention(hidden_size=32, num_heads=4, num_kv_heads=2).eval()
    hidden = torch.randn(2, 7, 32)
    angles = torch.randn(7, 8)
    rotary = (angles.cos(), angles.sin())
    mask = torch.ones(2, 1, 7, 7, dtype=torch.bool).tril() if mask_kind != "none" else None
    if mask_kind == "float":
        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float("-inf"))
    with torch.inference_mode():
        expected = attention(hidden, attention_mask=mask, image_rotary_emb=rotary)
        krea2_attention.configure_krea2_attention(attention, "cpu")
        assert not isinstance(attention.processor, krea2_attention.Krea2CudaAttnProcessor)
        krea2_attention.configure_krea2_attention(attention, "cuda:0")
        # Exerce sur CPU l'expansion qui évite le backend math sur CUDA.
        monkeypatch.setattr(krea2_attention, "needs_kv_expansion", lambda *args: True)
        sdpa, heads = torch.nn.functional.scaled_dot_product_attention, []
        def watched(query, key, value, **kwargs):
            heads.append((query.shape[1], key.shape[1], value.shape[1], kwargs["enable_gqa"]))
            return sdpa(query, key, value, **kwargs)
        monkeypatch.setattr(krea2_attention.F, "scaled_dot_product_attention", watched)
        actual = attention(hidden, attention_mask=mask, image_rotary_emb=rotary)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert heads == [(4, 4, 4, False)]


@pytest.mark.parametrize("flash,cudnn,expected", [(False, False, True), (True, False, False), (False, True, False)])
def test_masked_gqa_dispatch_checks_fused_kernel_capabilities(runtime, monkeypatch, flash, cudnn, expected):
    torch = runtime
    from pulid_app.models.krea2_attention import needs_kv_expansion
    cuda = torch.backends.cuda
    monkeypatch.setattr(cuda, "SDPAParams", lambda *args: args)
    monkeypatch.setattr(cuda, "flash_sdp_enabled", lambda: True)
    monkeypatch.setattr(cuda, "cudnn_sdp_enabled", lambda: True)
    monkeypatch.setattr(cuda, "can_use_flash_attention", lambda params: flash)
    monkeypatch.setattr(cuda, "can_use_cudnn_attention", lambda params: cudnn)
    query = SimpleNamespace(device=torch.device("cuda"), shape=(1, 4, 7, 8))
    key = SimpleNamespace(shape=(1, 2, 7, 8))
    assert needs_kv_expansion(query, key, key, object()) == expected
    assert not needs_kv_expansion(query, key, key, None)
    assert not needs_kv_expansion(query, query, query, object())
    query.device = torch.device("cpu")
    assert not needs_kv_expansion(query, key, key, object())


@pytest.mark.parametrize("fail", [False, True])
def test_progress_logs_phases_and_removes_hooks_even_on_failure(runtime, caplog, fail):
    torch = runtime
    from pulid_app.pipeline.krea2 import generation_progress
    pipe = SimpleNamespace(text_encoder=torch.nn.Identity(), transformer=torch.nn.Identity())
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    kwargs = {"latents": torch.ones(1)}
    try:
        with generation_progress(pipe, torch, "cpu", 2) as callback:
            pipe.text_encoder(kwargs["latents"])
            pipe.transformer(kwargs["latents"])
            assert callback(pipe, 0, 1000, kwargs) is kwargs
            if fail:
                raise RuntimeError("generation failed")
            assert callback(pipe, 1, 500, kwargs) is kwargs
    except RuntimeError:
        assert fail
    for module in (pipe.text_encoder, pipe.transformer):
        assert not module._forward_pre_hooks and not module._forward_hooks
    assert "prompt encodé" in caplog.text
    assert "step 1/2" in caplog.text
    assert ("décodage VAE" in caplog.text) == (not fail)


@pytest.mark.gpu
def test_real_cuda_masked_attention_without_math_backend(runtime):
    torch = runtime
    if not torch.cuda.is_available():
        pytest.skip("CUDA requis pour vérifier l'attention fusionnée masquée")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from diffusers.models.transformers.transformer_krea2 import Krea2Attention
    from pulid_app.models.krea2_attention import configure_krea2_attention
    attention = Krea2Attention(hidden_size=256, num_heads=4, num_kv_heads=2).to("cuda", dtype=torch.float16)
    configure_krea2_attention(attention, "cuda")
    hidden = torch.randn(1, 32, 256, device="cuda", dtype=torch.float16)
    mask = torch.ones(1, 1, 32, 32, device="cuda", dtype=torch.bool).tril()
    with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION]):
        result = attention(hidden, attention_mask=mask)
    assert result.shape == hidden.shape and torch.isfinite(result).all()
