"""Poids Comfy NVFP4 / FP8 conservés compactés, calcul couche par couche.

Importé uniquement après la configuration des caches et le bootstrap torch.
Spécification : Comfy-Org/comfy-quants, docs/formats/nvfp4.md (E2M1,
nibble haut en premier, échelles cuBLAS 128 x 4). Aucun runtime ComfyUI.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from pulid_app.exceptions import ModelLoadError


@dataclass(frozen=True)
class QuantizedSpec:
    format: str
    shape: tuple[int, int]
    full_precision_matrix_mult: bool = False


@dataclass(frozen=True)
class WeightLayout:
    quantized: dict[str, QuantizedSpec]
    auxiliary: frozenset[str]


def inspect_layout(weights: Any, path: Path, *, allow_quantized: bool) -> WeightLayout:
    """Valide les descripteurs et petites échelles, sans décompacter les poids."""
    keys = set(weights.keys())
    declarations: dict[str, dict[str, Any]] = {}
    auxiliary: set[str] = set()

    def fail(reason: str) -> None:
        raise ModelLoadError(
            f"Poids quantifiés incompatibles dans {path} : {reason}. "
            "Fournissez un fichier BF16/FP16/FP32, NVFP4 Comfy ou FP8 E4M3 scaled "
            "avec ses métadonnées et échelles complètes ; INT8/ConvRot ne sont pas pris en charge."
        )

    def declare(weight: str, config: Any) -> None:
        if weight not in keys or not isinstance(config, dict):
            fail(f"déclaration de couche invalide : {weight}")
        if set(config) - {"format", "full_precision_matrix_mult"}:
            fail(f"options de quantification inconnues : {weight}, {list(config)}")
        if config.get("format") not in {"nvfp4", "float8_e4m3fn"}:
            fail(f"format inconnu : {weight}, {config.get('format')!r}")
        if type(config.get("full_precision_matrix_mult", False)) is not bool:
            fail(f"full_precision_matrix_mult invalide : {weight}")
        previous = declarations.get(weight)
        if previous is not None and previous != config:
            fail(f"déclarations contradictoires : {weight}")
        declarations[weight] = config

    metadata = weights.metadata() or {}
    if "_quantization_metadata" in metadata:
        try:
            description = json.loads(metadata["_quantization_metadata"])
        except (ValueError, TypeError) as exc:
            raise ModelLoadError(f"Métadonnées de quantification JSON invalides dans {path}.") from exc
        if not isinstance(description, dict) or description.get("format_version") != "1.0" or not isinstance(description.get("layers"), dict):
            fail("_quantization_metadata doit décrire des layers au format_version 1.0")
        for layer, config in description["layers"].items():
            # Le checkpoint utilisateur préfixe les poids, mais pas les métadonnées.
            canonical = layer.removeprefix("model.diffusion_model.") + ".weight"
            matches = [key for key in keys if key.removeprefix("model.diffusion_model.") == canonical]
            if len(matches) != 1:
                fail(f"couche absente ou ambiguë dans les métadonnées : {layer}")
            declare(matches[0], config)

    for key in sorted(keys):
        if not key.endswith(".comfy_quant"):
            continue
        marker = weights.get_slice(key)
        if marker.get_dtype() != "U8" or len(marker.get_shape()) != 1 or not 0 < math.prod(marker.get_shape()) <= 4096:
            fail(f"marqueur JSON invalide : {key}")
        try:
            config = json.loads(bytes(weights.get_tensor(key).tolist()))
        except (ValueError, TypeError) as exc:
            raise ModelLoadError(f"Marqueur de quantification JSON invalide dans {path} : {key}.") from exc
        declare(key.removesuffix("comfy_quant") + "weight", config)
        auxiliary.add(key)

    def check(key: str, dtypes: set[str], shapes: set[tuple[int, ...]]) -> None:
        if key not in keys:
            fail(f"échelle absente : {key}")
        tensor = weights.get_slice(key)
        if tensor.get_dtype() not in dtypes or tuple(tensor.get_shape()) not in shapes:
            fail(f"type/dimensions invalides : {key}, {tensor.get_dtype()}, {tensor.get_shape()}")
        auxiliary.add(key)

    quantized: dict[str, QuantizedSpec] = {}
    for key, config in declarations.items():
        if not allow_quantized:
            fail(f"quantification interdite pour ce composant : {key}")
        tensor = weights.get_slice(key)
        shape = tuple(tensor.get_shape())
        if len(shape) != 2 or min(shape) <= 0:
            fail(f"matrice 2D requise : {key}")
        prefix = key.removesuffix("weight")
        scalar_keys = [prefix + "weight_scale"]
        if config["format"] == "nvfp4":
            if tensor.get_dtype() != "U8" or shape[1] % 8:
                fail(f"poids NVFP4 U8 avec blocs de 16 requis : {key}")
            shape = (shape[0], shape[1] * 2)
            scale_shape = (((shape[0] + 127) // 128) * 128, ((shape[1] // 16 + 3) // 4) * 4)
            check(prefix + "weight_scale", {"F8_E4M3"}, {scale_shape})
            scalar_keys = [prefix + "weight_scale_2"]
        elif tensor.get_dtype() != "F8_E4M3":
            fail(f"poids FP8 E4M3 requis : {key}")
        if prefix + "input_scale" in keys:
            scalar_keys.append(prefix + "input_scale")
        for scale_key in scalar_keys:
            check(scale_key, {"F32"}, {(), (1,)})
            value = weights.get_tensor(scale_key).item()
            if not math.isfinite(value) or value < 0 or (scale_key.endswith("input_scale") and value == 0):
                fail(f"échelle numérique invalide : {scale_key}")
        quantized[key] = QuantizedSpec(config["format"], shape, config.get("full_precision_matrix_mult", False))

    for key in keys - auxiliary - quantized.keys():
        dtype = weights.get_slice(key).get_dtype()
        if dtype not in {"F16", "BF16", "F32"} or any(part in key for part in ("scale_weight", "weight_scale", "input_scale", "pre_quant_scale", "scaled_fp8", "quantization")):
            fail(f"tenseur non déclaré ou format non pris en charge : {key}, {dtype}")
    return WeightLayout(quantized, frozenset(auxiliary))


def fp8_values(bits: torch.Tensor) -> torch.Tensor:
    """Décode E4M3FN en FP32 ; Metal ne sait pas stocker le dtype float8."""
    if bits.device.type != "mps":
        return bits.view(torch.float8_e4m3fn).float()
    codes = torch.arange(256, device=bits.device, dtype=torch.int32)
    exponent, mantissa = (codes >> 3) & 15, codes & 7
    table = torch.where(exponent == 0, mantissa.float() / 512, (1 + mantissa.float() / 8) * 2.0 ** (exponent - 7))
    table = torch.where((codes & 128) != 0, -table, table)
    table[(codes & 127) == 127] = float("nan")
    return table[bits.long()]


def dequantize_layer(
    packed: torch.Tensor, scale_bits: torch.Tensor, global_scale_bits: torch.Tensor | None,
    spec: QuantizedSpec, dtype: torch.dtype,
) -> torch.Tensor:
    """Une seule matrice temporaire ; les intermédiaires FP32 sont bornés par tranche."""
    rows, columns = spec.shape
    result = torch.empty(spec.shape, dtype=dtype, device=packed.device)
    if spec.format == "nvfp4":
        row_tiles, column_tiles = (rows + 127) // 128, (columns // 16 + 3) // 4
        # Adresse cuBLAS : [tuile_ligne, tuile_colonne, ligne%32, ligne//32, colonne%4].
        scales = fp8_values(scale_bits).reshape(row_tiles, column_tiles, 32, 4, 4)
        scales = scales.permute(0, 3, 2, 1, 4).reshape(row_tiles * 128, column_tiles * 4)
        scales = scales[:rows, :columns // 16] * global_scale_bits.view(torch.float32)
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=packed.device, dtype=torch.float32)
    else:
        scales = scale_bits.view(torch.float32)
    for start in range(0, rows, 512):
        end = min(start + 512, rows)
        data = packed[start:end]
        if spec.format == "nvfp4":
            codes = torch.stack((data >> 4, data & 15), dim=-1)
            values = levels[codes.long()].reshape(end - start, columns // 16, 16)
            values *= scales[start:end, :, None]
        else:
            values = fp8_values(data) * scales
        result[start:end].copy_(values.reshape(end - start, columns))
    return result


class QuantizedLinear(nn.Module):
    """Stockage compact même après .to(dtype), déplacement/offload via Accelerate.

    Les paramètres non entraînables sont des octets : poids FP4/FP8 et échelles
    gardent exactement leurs bits lors des changements de précision du pipeline.
    Aucune matrice déquantifiée n'est conservée après forward.
    """
    def __init__(self, linear: nn.Linear, weights: Any, source: str, spec: QuantizedSpec) -> None:
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.spec = spec
        self.bias = linear.bias
        self._fp8_unavailable = False
        prefix = source.removesuffix("weight")
        for name in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
            key = prefix + name
            if key in weights.keys():
                value = weights.get_tensor(key)
                # Les scalaires doivent avoir un axe pour être vus comme des octets.
                bits = value.reshape(1).view(torch.uint8) if value.ndim == 0 else value.view(torch.uint8)
                self.register_parameter(name, nn.Parameter(bits, requires_grad=False))
            elif name != "weight":
                self.register_parameter(name, None)

    def _can_use_fp8(self, value: torch.Tensor) -> bool:
        return (
            self.spec.format == "float8_e4m3fn" and not self.spec.full_precision_matrix_mult
            and not self._fp8_unavailable and value.device.type == "cuda"
            and value.dtype in {torch.float16, torch.bfloat16}
            and self.in_features % 16 == 0 and self.out_features % 16 == 0
            and torch.cuda.get_device_capability(value.device) >= (8, 9)
        )

    def _fp8_linear(self, value: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(-1, self.in_features)
        count = flat.shape[0]
        scale = (self.input_scale.to(value.device).view(torch.float32) if self.input_scale is not None
                 else flat.float().abs().amax().clamp_min(1e-12).reshape(1) / 448)
        encoded = (flat.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        # cuBLAS demande un nombre de lignes multiple de 16.
        padding = (-count) % 16
        if padding:
            encoded = torch.cat((encoded, torch.zeros((padding, self.in_features), device=value.device, dtype=encoded.dtype)))
        output = torch._scaled_mm(
            encoded, packed.view(torch.float8_e4m3fn).t(), scale_a=scale,
            scale_b=self.weight_scale.to(value.device).view(torch.float32),
            out_dtype=value.dtype, use_fast_accum=False,
        )
        return output[:count].reshape(*value.shape[:-1], self.out_features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        packed = self.weight.to(value.device)
        bias = None if self.bias is None else self.bias.to(device=value.device, dtype=value.dtype)
        if self._can_use_fp8(value):
            try:
                output = self._fp8_linear(value, packed)
                return output if bias is None else output + bias
            except torch.OutOfMemoryError:
                raise
            except (RuntimeError, NotImplementedError) as exc:
                self._fp8_unavailable = True
                logging.getLogger(__name__).debug("FP8 natif indisponible, repli par couche : %s", exc)
        weight = dequantize_layer(
            packed, self.weight_scale.to(value.device),
            None if self.weight_scale_2 is None else self.weight_scale_2.to(value.device),
            self.spec, value.dtype,
        )
        return F.linear(value, weight, bias)
