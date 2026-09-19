"""Fixtures synthétiques : aucun réseau, modèle téléchargé ni GPU obligatoire."""
from __future__ import annotations

import json
from types import SimpleNamespace
import weakref

import pytest

from pulid_app.exceptions import ModelLoadError
from pulid_app.models.krea2 import configure_krea2_memory, inspect_weight_format, load_safetensors_module
from pulid_app.paths import configure_external_model_caches


@pytest.fixture
def runtime(tmp_path):
    configure_external_model_caches(tmp_path)
    import torch
    return torch


def nvfp4_tensors(torch, prefix, rows=129, columns=80):
    """Encodeur de fixture par adresse scalaire, indépendant du décodeur tensoriel."""
    codes = torch.arange(rows * columns).reshape(rows, columns) % 16
    packed = ((codes[:, ::2] << 4) | codes[:, 1::2]).to(torch.uint8)
    blocks = columns // 16
    scales = 2.0 ** ((torch.arange(rows)[:, None] * 3 + torch.arange(blocks)) % 7 - 3)
    padded_rows, padded_columns = ((rows + 127) // 128) * 128, ((blocks + 3) // 4) * 4
    swizzled = torch.zeros(padded_rows * padded_columns)
    for row in range(rows):
        for column in range(blocks):
            tile = (row // 128) * (padded_columns // 4) + column // 4
            address = tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + column % 4
            swizzled[address] = scales[row, column]
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])
    global_scale = torch.tensor(0.03125)
    expected = levels[codes] * scales.repeat_interleave(16, dim=1) * global_scale
    return {
        prefix + "weight": packed,
        prefix + "weight_scale": swizzled.reshape(padded_rows, padded_columns).to(torch.float8_e4m3fn),
        prefix + "weight_scale_2": global_scale,
    }, expected


def fp8_tensors(torch, prefix, rows=16, columns=32, *, full_precision=False):
    data = ((torch.arange(rows * columns).reshape(rows, columns) % 17 - 8) / 4).to(torch.float8_e4m3fn)
    scale = torch.tensor(.125)
    marker = json.dumps({"format": "float8_e4m3fn", "full_precision_matrix_mult": full_precision}).encode()
    return {
        prefix + "weight": data, prefix + "weight_scale": scale,
        prefix + "comfy_quant": torch.tensor(list(marker), dtype=torch.uint8),
    }, data.float() * scale


def save(torch, path, tensors, *, nv_layer=None):
    from safetensors.torch import save_file
    metadata = None if nv_layer is None else {"_quantization_metadata": json.dumps({
        "format_version": "1.0", "layers": {nv_layer: {"format": "nvfp4"}},
    })}
    save_file(tensors, str(path), metadata=metadata)


@pytest.mark.parametrize("format", ["nvfp4", "float8_e4m3fn"])
@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_packed_storage_decode_numerics_and_temporary_lifetime(runtime, tmp_path, monkeypatch, format, dtype_name):
    torch = runtime
    from pulid_app.models import quantized
    path = tmp_path / "synthetic.safetensors"
    prefix = "model.diffusion_model.gate."
    tensors, dense = nvfp4_tensors(torch, prefix) if format == "nvfp4" else fp8_tensors(torch, prefix)
    rows, columns = dense.shape
    tensors[prefix + "bias"] = torch.linspace(-.2, .2, rows)
    save(torch, path, tensors, nv_layer="gate" if format == "nvfp4" else None)
    refs = []
    decode = quantized.dequantize_layer
    def watched(*args):
        matrix = decode(*args)
        refs.append(weakref.ref(matrix))
        return matrix
    monkeypatch.setattr(quantized, "dequantize_layer", watched)
    with torch.device("meta"):
        model = torch.nn.ModuleDict({"gate": torch.nn.Linear(columns, rows)})
    dtype = getattr(torch, dtype_name)
    model = load_safetensors_module(model, path, lambda k: k.removeprefix("model.diffusion_model."), dtype=dtype)
    assert refs == [], "Aucune déquantification pendant le chargement"
    layer = model["gate"]
    assert isinstance(layer, quantized.QuantizedLinear)
    before = {name: p.clone() for name, p in layer.named_parameters() if name != "bias"}
    layer.to(dtype=dtype)
    for name, original in before.items():
        assert getattr(layer, name).dtype == torch.uint8
        assert torch.equal(getattr(layer, name), original)
    x = torch.linspace(-1, 1, 2 * columns).reshape(2, columns).to(dtype)
    expected = torch.nn.functional.linear(x, dense.to(dtype), tensors[prefix + "bias"].to(dtype))
    with torch.inference_mode():
        for _ in range(2):
            torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
    assert len(refs) == 2 and all(ref() is None for ref in refs)
    assert layer.weight.numel() == rows * columns // (2 if format == "nvfp4" else 1)


@pytest.mark.parametrize("problem", ["missing_scale", "bad_scale_shape", "nan_scale", "undeclared", "unknown_format", "bad_marker", "pre_quant_scale", "bad_version", "conflict"])
def test_invalid_quantization_fails_before_loading(runtime, tmp_path, problem):
    torch = runtime
    from safetensors.torch import save_file
    path = tmp_path / "bad.safetensors"
    tensors, _ = nvfp4_tensors(torch, "gate.", rows=16, columns=32)
    metadata = {"format_version": "1.0", "layers": {"gate": {"format": "nvfp4"}}}
    if problem == "missing_scale":
        del tensors["gate.weight_scale_2"]
    elif problem == "bad_scale_shape":
        tensors["gate.weight_scale"] = torch.ones(16, 2).to(torch.float8_e4m3fn)
    elif problem == "nan_scale":
        tensors["gate.weight_scale_2"] = torch.tensor(float("nan"))
    elif problem == "undeclared":
        metadata["layers"] = {}
    elif problem == "unknown_format":
        metadata["layers"]["gate"]["format"] = "convrot_w4a4"
    elif problem == "bad_marker":
        tensors["gate.comfy_quant"] = torch.tensor([123], dtype=torch.uint8)
    elif problem == "pre_quant_scale":
        tensors["gate.pre_quant_scale"] = torch.ones(32)
    elif problem == "bad_version":
        metadata["format_version"] = "2.0"
    elif problem == "conflict":
        tensors["gate.comfy_quant"] = torch.tensor(list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8)
    save_file(tensors, str(path), metadata={"_quantization_metadata": json.dumps(metadata)})
    with torch.device("meta"):
        module = torch.nn.ModuleDict({"gate": torch.nn.Linear(32, 16, bias=False)})
    with pytest.raises(ModelLoadError) as error:
        load_safetensors_module(module, path, lambda k: k, dtype=torch.float32)
    assert str(path) in str(error.value)
    assert module["gate"].weight.device.type == "meta"


def test_quantized_shape_and_missing_regular_weight_remain_strict(runtime, tmp_path):
    torch = runtime
    path = tmp_path / "strict.safetensors"
    tensors, _ = fp8_tensors(torch, "gate.")
    save(torch, path, tensors)
    for columns, match in ((31, "Dimensions"), (32, "incomplets")):
        with torch.device("meta"):
            module = torch.nn.ModuleDict({"gate": torch.nn.Linear(columns, 16, bias=True)})
        with pytest.raises(ModelLoadError, match=match):
            load_safetensors_module(module, path, lambda k: k, dtype=torch.float32)
    with pytest.raises(ModelLoadError, match="interdite"):
        inspect_weight_format(path)  # Le VAE reste en précision flottante.


def test_nvfp4_decoding_spans_chunks_and_preserves_small_scales(runtime):
    torch = runtime
    from pulid_app.models.quantized import QuantizedSpec, dequantize_layer
    tensors, dense = nvfp4_tensors(torch, "gate.", rows=513, columns=80)
    tiny_scale = torch.tensor(1e-8)
    expected = (dense / .03125 * tiny_scale).to(torch.float16)
    result = dequantize_layer(tensors["gate.weight"],
        tensors["gate.weight_scale"].view(torch.uint8), tiny_scale.reshape(1).view(torch.uint8),
        QuantizedSpec("nvfp4", tuple(dense.shape)), torch.float16)
    assert result.count_nonzero() > 0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def load_fp8_layer(torch, tmp_path, *, full_precision=False):
    path = tmp_path / "fp8.safetensors"
    tensors, dense = fp8_tensors(torch, "gate.", full_precision=full_precision)
    save(torch, path, tensors)
    with torch.device("meta"):
        module = torch.nn.ModuleDict({"gate": torch.nn.Linear(32, 16, bias=False)})
    module = load_safetensors_module(module, path, lambda k: k, dtype=torch.float32)
    return module["gate"], dense


def test_accelerate_offloads_only_compact_parameters(runtime, tmp_path):
    torch = runtime
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module
    layer, dense = load_fp8_layer(torch, tmp_path)
    cpu_offload(layer, execution_device=torch.device("cpu"))
    assert all(p.device.type == "meta" for p in layer.parameters())
    value = torch.ones(1, 32)
    with torch.inference_mode():
        for _ in range(2):
            torch.testing.assert_close(layer(value), value @ dense.t())
            assert all(p.device.type == "meta" for p in layer.parameters())
    remove_hook_from_module(layer, recurse=True)
    assert layer.weight.dtype == torch.uint8 and layer.weight.device.type == "cpu"


def test_native_fp8_path_pads_inputs_without_dequantizing_weights(runtime, tmp_path, monkeypatch):
    torch = runtime
    from pulid_app.models import quantized
    layer, dense = load_fp8_layer(torch, tmp_path)
    monkeypatch.setattr(layer, "_can_use_fp8", lambda value: True)
    monkeypatch.setattr(quantized, "dequantize_layer", lambda *a: pytest.fail("native FP8 must keep compact weights"))
    calls = []
    def scaled_mm(a, b, *, scale_a, scale_b, out_dtype, use_fast_accum):
        calls.append((a.shape, b.shape, a.dtype, b.dtype))
        return ((a.float() * scale_a) @ (b.float() * scale_b)).to(out_dtype)
    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    x = torch.ones(1, 3, 32)
    torch.testing.assert_close(layer(x), x @ dense.t())
    assert calls == [(torch.Size([16, 32]), torch.Size([32, 16]), torch.float8_e4m3fn, torch.float8_e4m3fn)]


def test_fp8_fallback_is_local_and_does_not_swallow_oom(runtime, tmp_path, monkeypatch):
    torch = runtime
    layer, dense = load_fp8_layer(torch, tmp_path)
    monkeypatch.setattr(layer, "_can_use_fp8", lambda value: not layer._fp8_unavailable)
    calls = []
    def unavailable(*args):
        calls.append(True)
        raise RuntimeError("kernel not supported")
    monkeypatch.setattr(layer, "_fp8_linear", unavailable)
    x = torch.ones(1, 32)
    for _ in range(2):
        torch.testing.assert_close(layer(x), x @ dense.t())
    assert len(calls) == 1 and layer.weight.dtype == torch.uint8
    layer._fp8_unavailable = False
    def oom(*args):
        raise torch.OutOfMemoryError("CUDA OOM")
    monkeypatch.setattr(layer, "_fp8_linear", oom)
    with pytest.raises(torch.OutOfMemoryError):
        layer(x)


def test_fp8_dispatch_respects_hardware_and_full_precision_marker(runtime, tmp_path, monkeypatch):
    torch = runtime
    layer, _ = load_fp8_layer(torch, tmp_path)
    value = SimpleNamespace(device=torch.device("cuda"), dtype=torch.float16)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    assert layer._can_use_fp8(value)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 6))
    assert not layer._can_use_fp8(value)
    layer, _ = load_fp8_layer(torch, tmp_path, full_precision=True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    assert not layer._can_use_fp8(value)


@pytest.mark.parametrize("free_gib,offload,expected", [(1, "model_cpu_offload", "sequential"), (12, "model_cpu_offload", "model"), (1, "none", "to")])
def test_cuda_memory_policy_keeps_compact_model_or_offloads_layers(runtime, tmp_path, monkeypatch, free_gib, offload, expected):
    torch = runtime
    layer, _ = load_fp8_layer(torch, tmp_path)
    calls = []
    pipe = SimpleNamespace(transformer=layer, text_encoder=layer, vae=torch.nn.Linear(1, 1),
        to=lambda device: calls.append("to"),
        enable_model_cpu_offload=lambda **kw: calls.append("model"),
        enable_sequential_cpu_offload=lambda **kw: calls.append("sequential"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free_gib * 1024**3, 12 * 1024**3))
    configure_krea2_memory(pipe, device="cuda:0", offload=offload, dtype=torch.float16)
    assert calls == [expected]


@pytest.mark.parametrize("offload", [False, True])
def test_tiny_quantized_qwen_and_krea_generate_in_memory(runtime, tmp_path, offload):
    """Vraies architectures, débruitage et VAE, avec petits poids synthétiques."""
    torch = runtime
    from accelerate import cpu_offload
    from accelerate.hooks import remove_hook_from_module
    from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
    from safetensors.torch import save_file
    from pulid_app.models.krea2 import workflow_pipeline, text_encoder_key, transformer_key
    from pulid_app.models.quantized import QuantizedLinear

    transformer = Krea2Transformer2DModel(
        num_layers=1, attention_head_dim=8, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, timestep_embed_dim=16, text_hidden_dim=16, num_text_layers=2,
        text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=32,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(2, 2, 4),
    )
    state, layers = dict(transformer.state_dict()), {}
    for name, layer in transformer.named_modules():
        if name.startswith("transformer_blocks.") and isinstance(layer, torch.nn.Linear):
            tensors, _ = nvfp4_tensors(torch, name + ".", layer.out_features, layer.in_features)
            state.update(tensors)
            layers[name] = {"format": "nvfp4"}
    path = tmp_path / "krea.safetensors"
    save_file(state, str(path), metadata={"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})})
    with torch.device("meta"):
        empty = Krea2Transformer2DModel.from_config(transformer.config)
    transformer = load_safetensors_module(empty, path, transformer_key, dtype=torch.float32, keep_norm_fp32=True)

    config = Qwen3VLTextConfig(hidden_size=16, intermediate_size=32, num_attention_heads=2,
        num_key_value_heads=2, head_dim=8, num_hidden_layers=2, vocab_size=16, use_cache=False)
    text = Qwen3VLTextModel(config)
    state = {"model." + name: value for name, value in text.state_dict().items()}
    for name, layer in text.named_modules():
        if isinstance(layer, torch.nn.Linear):
            tensors, _ = fp8_tensors(torch, "model." + name + ".", layer.out_features, layer.in_features)
            state.update(tensors)
    path = tmp_path / "qwen.safetensors"
    save_file(state, str(path))
    # init_empty_weights préserve les buffers RoPE non persistants sur CPU.
    from accelerate import init_empty_weights
    with init_empty_weights():
        text = Qwen3VLTextModel(config)
    text = load_safetensors_module(text, path, text_encoder_key, dtype=torch.float32)
    vae = AutoencoderKLQwenImage(base_dim=4, dim_mult=[1, 2, 2, 2], num_res_blocks=1)
    pipe = workflow_pipeline(transformer=transformer, text_encoder=text, tokenizer=None,
        vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        is_distilled=True, text_encoder_select_layers=(1, 2))
    pipe.set_progress_bar_config(disable=True)
    if offload:
        # Même mécanisme Accelerate que CUDA, exercé sur CPU sans carte requise.
        for model in (transformer, text, vae):
            cpu_offload(model, execution_device=torch.device("cpu"))
    try:
        before = sorted(tmp_path.rglob("*"))
        with torch.inference_mode():
            encoded = text(input_ids=torch.tensor([[1, 2, 3, 4]]), output_hidden_states=True)
            embeds = torch.stack(encoded.hidden_states[1:], dim=2)
            assert torch.isfinite(embeds).all()
            result = pipe(prompt_embeds=embeds, prompt_embeds_mask=torch.ones(1, 4, dtype=torch.bool),
                width=64, height=64, num_inference_steps=2, sigmas=[1., .5], guidance_scale=0,
                generator=torch.Generator().manual_seed(42), output_type="np")
        assert result.images.shape == (1, 64, 64, 3)
        assert torch.isfinite(torch.from_numpy(result.images)).all()
        assert sorted(tmp_path.rglob("*")) == before
        for model in (transformer, text):
            assert any(isinstance(layer, QuantizedLinear) for layer in model.modules())
            assert all(layer.weight.dtype == torch.uint8 for layer in model.modules() if isinstance(layer, QuantizedLinear))
    finally:
        pipe.remove_all_hooks()
        for model in (transformer, text, vae):
            remove_hook_from_module(model, recurse=True)
        assert all(p.device.type == "cpu" for model in (transformer, text, vae) for p in model.parameters())


@pytest.mark.gpu
def test_real_cuda_nvfp4_layer_when_available(runtime, tmp_path):
    torch = runtime
    if not torch.cuda.is_available():
        pytest.skip("CUDA requis pour vérifier la déquantification NVFP4 sur GPU")
    tensors, dense = nvfp4_tensors(torch, "gate.")
    path = tmp_path / "cuda-nvfp4.safetensors"
    save(torch, path, tensors, nv_layer="gate")
    with torch.device("meta"):
        module = torch.nn.ModuleDict({"gate": torch.nn.Linear(dense.shape[1], dense.shape[0], bias=False)})
    module = load_safetensors_module(module, path, lambda k: k, dtype=torch.float16).to("cuda")
    value = torch.ones(1, dense.shape[1], device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        actual = module["gate"](value)
    torch.testing.assert_close(actual, value @ dense.to(device="cuda", dtype=value.dtype).t())
    assert module["gate"].weight.dtype == torch.uint8


@pytest.mark.gpu
def test_real_cuda_fp8_kernel_when_available(runtime, tmp_path):
    torch = runtime
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("CUDA Ada ou plus récent requis pour le calcul FP8 natif")
    layer, dense = load_fp8_layer(torch, tmp_path)
    layer.to("cuda")
    value = torch.ones(1, 3, 32, device="cuda", dtype=torch.float16)
    # Appel direct : un échec du noyau ne doit pas être caché par le repli.
    with torch.inference_mode():
        actual = layer._fp8_linear(value, layer.weight)
    torch.testing.assert_close(actual, value @ dense.to(device="cuda", dtype=value.dtype).t(), rtol=.01, atol=.01)


@pytest.mark.gpu
def test_mps_compact_weights_when_available(runtime, tmp_path):
    torch = runtime
    if not torch.backends.mps.is_available():
        pytest.skip("Metal requis pour vérifier le décodeur E4M3 portable")
    from pulid_app.models.quantized import dequantize_layer, QuantizedSpec
    tensors, expected = nvfp4_tensors(torch, "gate.")
    result = dequantize_layer(tensors["gate.weight"].to("mps"),
        tensors["gate.weight_scale"].view(torch.uint8).to("mps"),
        tensors["gate.weight_scale_2"].reshape(1).view(torch.uint8).to("mps"),
        QuantizedSpec("nvfp4", tuple(expected.shape)), torch.float32)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    layer, dense = load_fp8_layer(torch, tmp_path)
    layer.to("mps")
    value = torch.ones(1, 32, device="mps")
    with torch.inference_mode():
        torch.testing.assert_close(layer(value).cpu(), value.cpu() @ dense.t())
