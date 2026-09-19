#!/usr/bin/env python3
"""Mesure le décodage NVFP4 ou l'attention Krea CUDA, sans modèle ni fichier créé."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def benchmark_attention(torch, device, runs: int) -> dict:
    """Dimensions de la diffusion 1248×832 avec la fenêtre texte du pipeline."""
    from pulid_app.models.krea2_attention import krea2_sdpa
    from pulid_app.pipeline.krea2 import KREA2_TEXT_SEQUENCE_LENGTH

    tokens, heads, kv_heads, head_dim = 4056 + KREA2_TEXT_SEQUENCE_LENGTH, 48, 12, 128
    # Même disposition non contiguë après projection/transposition que Diffusers.
    query = torch.randn(1, tokens, heads, head_dim, device=device, dtype=torch.float16).transpose(1, 2)
    key, value = (torch.randn(1, tokens, kv_heads, head_dim, device=device,
                             dtype=torch.float16).transpose(1, 2) for _ in range(2))
    mask = torch.ones(1, 1, 1, tokens, device=device, dtype=torch.bool)
    mask[..., 64:KREA2_TEXT_SEQUENCE_LENGTH] = False
    with torch.inference_mode():
        result = krea2_sdpa(query, key, value, mask)
        if not torch.isfinite(result).all().item():
            raise RuntimeError("L'attention CUDA a produit des valeurs non finies.")
        del result
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        for _ in range(runs):
            result = krea2_sdpa(query, key, value, mask)
            del result
        torch.cuda.synchronize(device)
        elapsed_ms = 1000 * (perf_counter() - started) / runs
        peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
        # Trace des opérateurs ATen réellement appelés ; aucun fichier de profil.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            krea2_sdpa(query, key, value, mask)
            torch.cuda.synchronize(device)
    operators = sorted(event.key for event in profile.key_averages()
                       if event.key.startswith("aten::_scaled_dot_product"))
    if not operators or any("math" in operator for operator in operators):
        raise RuntimeError(f"Attention fusionnée non confirmée : {operators}")
    return {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "cuda": torch.version.cuda, "image_size": [1248, 832], "query_shape": list(query.shape),
        "kv_shape": list(key.shape), "runs": runs, "sdpa_operators": operators,
        "attention_ms": round(elapsed_ms, 3), "peak_allocated_mib": round(peak_mib, 1)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--size", type=int, default=6144)
    parser.add_argument("--attention", action="store_true",
                        help="Mesurer uniquement l'attention masquée à 1248×832 et identifier son noyau")
    args = parser.parse_args()
    if not 1 <= args.runs <= 20 or not 128 <= args.size <= 8192 or args.size % 128:
        parser.error("runs : 1..20 ; size : multiple de 128 entre 128 et 8192")
    from pulid_app.config import load_config
    from pulid_app.paths import configure_external_model_caches

    configure_external_model_caches(load_config(args.config).models_root)
    import torch
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        print("CUDA requis pour ce benchmark.", file=sys.stderr)
        return 1
    try:
        device = torch.device(args.device)
        if args.attention:
            print(json.dumps(benchmark_attention(torch, device, args.runs), indent=2))
            return 0
        from pulid_app.models.quantized import QuantizedSpec, dequantize_layer
        from pulid_app.models.quantized_cuda import CudaQuantizedKernels

        kernels = CudaQuantizedKernels()
        size = args.size
        packed = torch.full((size, size // 2), 0x24, dtype=torch.uint8, device=device)
        scales = torch.ones((size, size // 16), device=device).to(torch.float8_e4m3fn).view(torch.uint8)
        global_scale = torch.tensor([.03125], device=device).view(torch.uint8)
        spec = QuantizedSpec("nvfp4", (size, size))

        def measure(decoder):
            with torch.inference_mode():
                decoder(packed, scales, global_scale, spec, torch.float16)
                torch.cuda.synchronize(device)
                started = perf_counter()
                for _ in range(args.runs):
                    result = decoder(packed, scales, global_scale, spec, torch.float16)
                    del result
                torch.cuda.synchronize(device)
            return 1000 * (perf_counter() - started) / args.runs

        native = measure(kernels.dequantize)
        portable = measure(dequantize_layer)
        print(json.dumps({"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
            "cuda": torch.version.cuda, "shape": [size, size], "runs": args.runs,
            "backend": "comfy-kitchen CUDA", "native_decode_ms": round(native, 3),
            "portable_decode_ms": round(portable, 3), "decode_speedup": round(portable / native, 2)}, indent=2))
        return 0
    except (RuntimeError, OSError) as exc:
        print(f"Benchmark impossible : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
