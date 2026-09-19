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


@pytest.mark.parametrize("flash,cudnn,expected", [(False, False, True), (True, False, False), (False, True, True)])
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
    # Même repli pour float32/GQA sans masque si Flash n'est pas éligible.
    assert needs_kv_expansion(query, key, key, None) == expected
    assert not needs_kv_expansion(query, query, query, object())
    query.device = torch.device("cpu")
    assert not needs_kv_expansion(query, key, key, object())


@pytest.mark.parametrize("available,expected", [
    ((True, True, True), "FLASH_ATTENTION"),
    ((False, True, True), "EFFICIENT_ATTENTION"),
    ((False, False, True), "CUDNN_ATTENTION"),
    ((False, False, False), None),
])
@pytest.mark.parametrize("fail", [False, True])
def test_cuda_dispatch_excludes_math_and_restores_flags(runtime, monkeypatch, caplog, available, expected, fail):
    torch = runtime
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from pulid_app.exceptions import GenerationError
    from pulid_app.models.krea2_attention import krea2_sdpa
    cuda = torch.backends.cuda
    monkeypatch.setattr(cuda, "SDPAParams", lambda *args: args)
    for name, supported in zip(("flash", "efficient", "cudnn"), available):
        monkeypatch.setattr(cuda, f"can_use_{name}_attention", lambda params, supported=supported: supported)
    query = SimpleNamespace(device=torch.device("cuda"), dtype=torch.float16, shape=(1, 48, 4568, 128))
    kv = SimpleNamespace(shape=(1, 12, 4568, 128))
    expanded = []
    def repeat(repeats, dim):
        assert repeats == 4 and dim == 1
        expanded.append(True)
        return query
    kv.repeat_interleave = repeat
    mask = object()
    calls = []
    def run(q, k, v, **kwargs):
        assert q is query and kwargs["attn_mask"] is mask
        assert kwargs["enable_gqa"] == available[0]
        assert k is v is (kv if available[0] else query)
        assert not cuda.math_sdp_enabled()
        assert cuda.flash_sdp_enabled() == (expected == "FLASH_ATTENTION")
        assert cuda.mem_efficient_sdp_enabled() == (expected == "EFFICIENT_ATTENTION")
        assert cuda.cudnn_sdp_enabled() == (expected == "CUDNN_ATTENTION")
        calls.append(True)
        if fail:
            raise torch.OutOfMemoryError("test OOM")
        return "result"
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", run)
    # Ordre Ada : math avant cuDNN. L'ancien chemin n'interdisait pas ce repli.
    order = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH,
             SDPBackend.CUDNN_ATTENTION]
    reported = set()
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    with sdpa_kernel(order, set_priority=True):
        priority = torch._C._get_sdp_priority_order()
        if expected is None:
            with pytest.raises(GenerationError, match="benchmark_krea2_cuda.py --attention"):
                krea2_sdpa(query, kv, kv, mask, reported=reported)
            assert not calls
        elif fail:
            with pytest.raises(torch.OutOfMemoryError, match="test OOM"):
                krea2_sdpa(query, kv, kv, mask, reported=reported)
        else:
            for _ in range(2):
                assert krea2_sdpa(query, kv, kv, mask, reported=reported) == "result"
            assert caplog.text.count("Krea 2 : attention CUDA") == 1
            assert expected in caplog.text
        assert cuda.math_sdp_enabled() and cuda.flash_sdp_enabled()
        assert cuda.mem_efficient_sdp_enabled() and cuda.cudnn_sdp_enabled()
        assert torch._C._get_sdp_priority_order() == priority
    assert bool(expanded) == (not available[0])


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
@pytest.mark.parametrize("mask_kind", ["padding", "additive", "none"])
def test_real_cuda_attention_uses_fused_operator_with_math_enabled(runtime, mask_kind):
    torch = runtime
    if not torch.cuda.is_available():
        pytest.skip("CUDA requis pour vérifier l'attention fusionnée masquée")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from pulid_app.models.krea2_attention import krea2_sdpa
    query = torch.randn(1, 64, 48, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
    key, value = (torch.randn(1, 64, 12, 128, device="cuda", dtype=torch.float16).transpose(1, 2)
                  for _ in range(2))
    mask = torch.ones(1, 1, 1, 64, device="cuda", dtype=torch.bool)
    mask[..., 32:48] = False
    if mask_kind == "additive":
        mask = torch.zeros_like(mask, dtype=query.dtype).masked_fill(~mask, float("-inf"))
    elif mask_kind == "none":
        mask = None
    with torch.inference_mode():
        with sdpa_kernel(SDPBackend.MATH):
            expected = torch.nn.functional.scaled_dot_product_attention(query, key, value,
                attn_mask=mask, enable_gqa=True)
        # N'interdire PAS math dans le test : c'est le code applicatif qui doit
        # garantir ce choix, même avec math placé avant tous les noyaux fusionnés.
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION], set_priority=True):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                result = krea2_sdpa(query, key, value, mask)
                torch.cuda.synchronize()
            assert torch.backends.cuda.math_sdp_enabled()
    operators = {event.key for event in profile.key_averages()}
    assert "aten::_scaled_dot_product_attention_math" not in operators
    assert operators.intersection({"aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_flash_attention", "aten::_scaled_dot_product_cudnn_attention"})
    torch.testing.assert_close(result, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.gpu
def test_real_cuda_attention_benchmark_at_workflow_resolution(runtime):
    torch = runtime
    if not torch.cuda.is_available():
        pytest.skip("CUDA requis pour le benchmark 1248×832")
    from scripts.benchmark_krea2_cuda import benchmark_attention
    result = benchmark_attention(torch, torch.device("cuda:0"), 1)
    assert result["query_shape"] == [1, 48, 5080, 128]
    assert result["attention_ms"] > 0 and result["sdpa_operators"]
    assert not any("math" in name for name in result["sdpa_operators"])
