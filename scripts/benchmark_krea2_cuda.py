#!/usr/bin/env python3
"""Compare les décodeurs NVFP4 sur une matrice synthétique, sans modèle ni fichier créé."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--size", type=int, default=6144)
    args = parser.parse_args()
    if not 1 <= args.runs <= 20 or not 128 <= args.size <= 8192 or args.size % 128:
        parser.error("runs : 1..20 ; size : multiple de 128 entre 128 et 8192")
    from pulid_app.config import load_config
    from pulid_app.paths import configure_external_model_caches

    configure_external_model_caches(load_config(args.config).models_root)
    import torch
    from pulid_app.models.quantized import QuantizedSpec, dequantize_layer
    from pulid_app.models.quantized_cuda import CudaQuantizedKernels

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        print("CUDA requis pour ce benchmark.", file=sys.stderr)
        return 1
    try:
        kernels = CudaQuantizedKernels()
        size, device = args.size, torch.device(args.device)
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
