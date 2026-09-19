"""Accès aux noyaux CUDA Comfy Kitchen, sans import de ComfyUI."""
from __future__ import annotations

from typing import Any

from pulid_app.exceptions import ModelLoadError


class CudaQuantizedKernels:
    def __init__(self) -> None:
        try:
            import comfy_kitchen
            from comfy_kitchen.registry import registry
        except (ImportError, OSError) as exc:
            raise ModelLoadError(
                "Noyaux CUDA Krea absents ou inutilisables (comfy-kitchen==0.2.35). "
                "Lancez install_windows.bat --update pour installer les dépendances verrouillées, "
                "puis start_windows.bat --offload model_cpu_offload."
            ) from exc
        backend = comfy_kitchen.list_backends().get("cuda", {})
        if not backend.get("available") or backend.get("disabled"):
            raise ModelLoadError(
                f"Noyaux CUDA Krea indisponibles : {backend.get('unavailable_reason') or 'backend CUDA désactivé'}. "
                "Lancez install_windows.bat --update et utilisez start_windows.bat ; "
                "le décodage PyTorch lent n'est pas activé silencieusement sur CUDA."
            )
        self.registry = registry

    def dequantize(self, packed: Any, scale_bits: Any, global_scale_bits: Any, spec: Any, dtype: Any) -> Any:
        import torch

        if spec.format == "nvfp4":
            function = "dequantize_nvfp4"
            arguments = dict(qx=packed, per_tensor_scale=global_scale_bits.view(torch.float32),
                block_scales=scale_bits.view(torch.float8_e4m3fn), output_type=dtype, hi_first=True)
        else:
            function = "dequantize_per_tensor_fp8"
            arguments = dict(x=packed.view(torch.float8_e4m3fn), scale=scale_bits.view(torch.float32), output_type=dtype)
        # Un backend explicite interdit un repli implicite vers le code eager.
        kernel = self.registry.get_implementation(function, backend="cuda", kwargs=arguments)
        return kernel(**arguments)


def configure_cuda_kernels(models: tuple[Any, ...], device: str) -> None:
    if device.split(":")[0] != "cuda":
        return
    from pulid_app.models.quantized import QuantizedLinear

    layers = [layer for model in models for layer in model.modules() if isinstance(layer, QuantizedLinear)]
    if not layers:
        return
    kernels = CudaQuantizedKernels()
    for layer in layers:
        layer.cuda_kernels = kernels
