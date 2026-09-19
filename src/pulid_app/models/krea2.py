"""Chargement strict, local et sans identité des composants Krea 2.

Les noms natifs viennent de krea-ai/krea-2 (mmdit.py). Les calculs restent
ceux de Diffusers/Transformers ; aucun import ni runtime ComfyUI.
"""
from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import re
from typing import Any

from pulid_app.config import Krea2Config
from pulid_app.exceptions import ModelLoadError, ModelNotFoundError


def require_krea_file(path: Path, action: str) -> None:
    if not path.is_file():
        raise ModelNotFoundError(f"Fichier Krea 2 absent ou non régulier : {path}. {action}")
    try:
        with path.open("rb") as stream:
            if not stream.read(1):
                raise ModelLoadError(f"Fichier vide : {path}. {action}")
    except OSError as exc:
        raise ModelLoadError(f"Fichier illisible : {path}. Vérifiez les permissions. {action}") from exc


def validate_krea2_assets(config: Krea2Config) -> None:
    for path in (config.checkpoint, config.text_encoder):
        require_krea_file(path, "Déposez manuellement les poids BF16/FP16/FP32 indiqués dans la configuration.")
        if path.suffix.lower() != ".safetensors":
            raise ModelLoadError(f"Format attendu .safetensors : {path}.")
    require_krea_file(config.vae, "Relancez pulid-install --krea2-only pour installer le VAE.")
    directory = config.text_encoder_config_dir
    if not directory.is_dir():
        raise ModelNotFoundError(f"Configuration qwen3vl absente : {directory}. Déposez config.json et le tokenizer Qwen3-VL-4B-Instruct dans ce dossier.")
    for filename in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        require_krea_file(directory / filename, "Copiez manuellement les fichiers de configuration/tokenizer Qwen3-VL-4B-Instruct.")


def transformer_key(key: str) -> str:
    """Traduit une clé officielle Krea en clé Diffusers, sans modifier les poids."""
    key = key.removeprefix("model.diffusion_model.")
    prefixes = {
        "first.": "img_in.", "tmlp.0.": "time_embed.linear_1.",
        "tmlp.2.": "time_embed.linear_2.", "tproj.1.": "time_mod_proj.",
        "txtfusion.": "text_fusion.", "txtmlp.0.": "txt_in.norm.",
        "txtmlp.1.": "txt_in.linear_1.", "txtmlp.3.": "txt_in.linear_2.",
        "blocks.": "transformer_blocks.", "last.": "final_layer.",
    }
    for source, target in prefixes.items():
        if key.startswith(source):
            key = target + key[len(source):]
            break
    substitutions = {
        ".mod.lin": ".scale_shift_table", ".modulation.lin": ".scale_shift_table",
        ".prenorm.": ".norm1.", ".postnorm.": ".norm2.",
        ".attn.wq.": ".attn.to_q.", ".attn.wk.": ".attn.to_k.",
        ".attn.wv.": ".attn.to_v.", ".attn.wo.": ".attn.to_out.0.",
        ".attn.gate.": ".attn.to_gate.",
        ".attn.qknorm.qnorm.": ".attn.norm_q.",
        ".attn.qknorm.knorm.": ".attn.norm_k.",
        ".mlp.gate.": ".ff.gate.", ".mlp.up.": ".ff.up.",
        ".mlp.down.": ".ff.down.",
    }
    for source, target in substitutions.items():
        key = key.replace(source, target)
    if key.endswith(".scale"):
        key = key[:-6] + ".weight"
    return key


def text_encoder_key(key: str) -> str | None:
    key = key.removeprefix("text_encoders.qwen3vl_4b.transformer.")
    for prefix in ("model.language_model.", "language_model.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    # La branche vision et la tête de génération de texte ne sont pas utilisées.
    if key.startswith(("visual.", "lm_head.")):
        return None
    return key


def inspect_weight_format(path: Path) -> None:
    """Lit seulement l'en-tête avant toute allocation d'un gros modèle."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as weights:
        for key in weights.keys():
            dtype = weights.get_slice(key).get_dtype()
            if dtype not in {"F16", "BF16", "F32"} or re.search(r"(?:scale_weight|weight_scale|quantization|comfy_quant)", key):
                raise ModelLoadError(
                    f"Poids quantifiés non pris en charge dans {path} ({key}, {dtype}). "
                    "Fournissez manuellement une version BF16/FP16/FP32 ; les formats "
                    "ComfyUI FP8 scaled, INT8 et NVFP4 ne sont pas convertis implicitement."
                )


def load_safetensors_module(
    module: Any, path: Path, key_mapper: Callable[[str], str | None], *, dtype: Any,
    keep_norm_fp32: bool = False,
) -> Any:
    """Valide toutes les clés/shapes sur meta, puis charge un tenseur à la fois."""
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    import torch

    expected = module.state_dict()
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        mapping: dict[str, str] = {}
        for source in weights.keys():
            target = key_mapper(source)
            if target is None:
                continue
            if target not in expected or target in mapping:
                raise ModelLoadError(f"Clé incompatible ou dupliquée dans {path} : {source} → {target}.")
            shape = tuple(weights.get_slice(source).get_shape())
            wanted = tuple(expected[target].shape)
            # La modulation native est aplatie (6 * hidden_size).
            if shape != wanted and not (target.endswith(".scale_shift_table") and shape == (expected[target].numel(),)):
                raise ModelLoadError(f"Dimensions incompatibles dans {path} : {source}, {shape} au lieu de {wanted}.")
            mapping[target] = source
        missing = expected.keys() - mapping.keys()
        if missing:
            raise ModelLoadError(f"Poids incomplets dans {path} : {', '.join(sorted(missing)[:8])}. Fournissez le checkpoint complet compatible.")
        for target, source in mapping.items():
            value = weights.get_tensor(source).reshape(expected[target].shape)
            target_dtype = torch.float32 if keep_norm_fp32 and "norm" in target else dtype
            set_module_tensor_to_device(module, target, "cpu", value=value, dtype=target_dtype)
    return module.eval().requires_grad_(False)


def workflow_pipeline(**components: Any) -> Any:
    from diffusers import Krea2Pipeline

    class WorkflowKrea2Pipeline(Krea2Pipeline):
        @property
        def do_classifier_free_guidance(self) -> bool:
            # Diffusers utilise cond + guidance*(cond-uncond), donc cfg-1.
            return self.guidance_scale != 0

    return WorkflowKrea2Pipeline(**components)


def load_krea2_pipeline(config: Krea2Config, *, device: str, dtype: Any, offload: str) -> Any:
    """Assemble uniquement des composants locaux. Le bootstrap configure les caches."""
    validate_krea2_assets(config)
    try:
        import torch
        from accelerate import init_empty_weights
        from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3VLTextConfig, Qwen3VLTextModel

        for path in (config.checkpoint, config.text_encoder, config.vae):
            inspect_weight_format(path)
        raw = json.loads((config.text_encoder_config_dir / "config.json").read_text(encoding="utf-8"))
        text_config = Qwen3VLTextConfig(**raw.get("text_config", raw))
        if (text_config.hidden_size, text_config.num_hidden_layers) != (2560, 36):
            raise ModelLoadError(f"Qwen3-VL-4B requis : {config.text_encoder_config_dir / 'config.json'}.")
        text_config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(
            str(config.text_encoder_config_dir), local_files_only=True, trust_remote_code=False,
        )
        with init_empty_weights():
            transformer = Krea2Transformer2DModel()
        transformer = load_safetensors_module(transformer, config.checkpoint, transformer_key, dtype=dtype, keep_norm_fp32=True)
        # Qwen conserve ses buffers RoPE calculés sur CPU, même sous init_empty_weights.
        with init_empty_weights():
            text_encoder = Qwen3VLTextModel(text_config)
        text_encoder = load_safetensors_module(text_encoder, config.text_encoder, text_encoder_key, dtype=dtype)
        with init_empty_weights():
            vae = AutoencoderKLQwenImage()
        state = load_file(str(config.vae), device="cpu")
        if "encoder.conv1.weight" in state:
            state = convert_wan_vae_to_diffusers(state)
        vae.load_state_dict(state, strict=True, assign=True)
        vae = vae.to(dtype=torch.float32).eval().requires_grad_(False)
        del state

        pipeline = workflow_pipeline(
            transformer=transformer, text_encoder=text_encoder, tokenizer=tokenizer,
            vae=vae, scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
            is_distilled=True,
        )
        pipeline.vae.enable_tiling()
        if offload == "model_cpu_offload":
            pipeline.enable_model_cpu_offload(device=device)
        else:
            pipeline.to(device)
        return pipeline
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            f"Chargement Krea 2 impossible ({config.checkpoint}, {config.text_encoder}, {config.vae}) : {exc}. "
            "Vérifiez les fichiers locaux et réinstallez les dépendances d'inférence verrouillées."
        ) from exc
