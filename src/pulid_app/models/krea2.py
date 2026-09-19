"""Chargement strict, local et sans identité des composants Krea 2.

Les noms natifs viennent de krea-ai/krea-2 (mmdit.py). Les calculs restent
ceux de Diffusers/Transformers ; aucun import ni runtime ComfyUI.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pulid_app.config import Krea2Config
from pulid_app.exceptions import ModelLoadError, ModelNotFoundError

if TYPE_CHECKING:
    from pulid_app.models.quantized import WeightLayout


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
        require_krea_file(path, "Déposez manuellement les poids BF16/FP16/FP32 ou quantifiés compatibles indiqués dans la configuration.")
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


def inspect_weight_format(path: Path, *, allow_quantized: bool = False) -> WeightLayout:
    """Valide l'en-tête et les petits descripteurs avant allocation du modèle."""
    from safetensors import safe_open
    from pulid_app.models.quantized import inspect_layout

    with safe_open(str(path), framework="pt", device="cpu") as weights:
        return inspect_layout(weights, path, allow_quantized=allow_quantized)


def load_safetensors_module(
    module: Any, path: Path, key_mapper: Callable[[str], str | None], *, dtype: Any,
    keep_norm_fp32: bool = False,
) -> Any:
    """Valide toutes les clés/shapes sur meta, puis charge un tenseur à la fois."""
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    import torch
    from pulid_app.models.quantized import QuantizedLinear, inspect_layout

    expected = module.state_dict()
    with safe_open(str(path), framework="pt", device="cpu") as weights:
        layout = inspect_layout(weights, path, allow_quantized=True)
        mapping: dict[str, str] = {}
        for source in weights.keys():
            if source in layout.auxiliary:
                continue
            target = key_mapper(source)
            if target is None:
                continue
            if target not in expected or target in mapping:
                raise ModelLoadError(f"Clé incompatible ou dupliquée dans {path} : {source} → {target}.")
            spec = layout.quantized.get(source)
            shape = spec.shape if spec else tuple(weights.get_slice(source).get_shape())
            wanted = tuple(expected[target].shape)
            # La modulation native est aplatie (6 * hidden_size).
            if shape != wanted and not (target.endswith(".scale_shift_table") and shape == (expected[target].numel(),)):
                raise ModelLoadError(f"Dimensions incompatibles dans {path} : {source}, {shape} au lieu de {wanted}.")
            if spec and (target.rsplit(".", 1)[-1] != "weight" or not isinstance(module.get_submodule(target.rpartition(".")[0]), torch.nn.Linear)):
                raise ModelLoadError(f"La quantification de {path} : {source} exige une couche Linear compatible.")
            mapping[target] = source
        missing = expected.keys() - mapping.keys()
        if missing:
            raise ModelLoadError(f"Poids incomplets dans {path} : {', '.join(sorted(missing)[:8])}. Fournissez le checkpoint complet compatible.")
        # Remplacer seulement après validation complète, sans jamais créer une
        # copie BF16/FP16 de l'ensemble des matrices quantifiées.
        for target, source in mapping.items():
            if source in layout.quantized:
                location = target.rpartition(".")[0]
                replacement = QuantizedLinear(module.get_submodule(location), weights, source, layout.quantized[source])
                if location:
                    module.set_submodule(location, replacement)
                else:
                    module = replacement
        for target, source in mapping.items():
            if source in layout.quantized:
                continue
            value = weights.get_tensor(source).reshape(expected[target].shape)
            target_dtype = torch.float32 if keep_norm_fp32 and "norm" in target else dtype
            set_module_tensor_to_device(module, target, "cpu", value=value, dtype=target_dtype)
    if layout.quantized:
        logging.getLogger(__name__).info(
            "%s : poids %s conservés compactés, calcul/déquantification par couche.",
            path, ", ".join(sorted({spec.format for spec in layout.quantized.values()})),
        )
    return module.eval().requires_grad_(False)


def configure_krea2_memory(pipeline: Any, *, device: str, offload: str, dtype: Any) -> None:
    """Garde un composant compacté en VRAM s'il reste une marge pour le calcul."""
    import torch
    from pulid_app.models.quantized import QuantizedLinear

    if offload != "model_cpu_offload":
        pipeline.to(device)
        return
    models = (pipeline.transformer, pipeline.text_encoder, pipeline.vae)
    resident = max(sum(t.numel() * t.element_size() for t in model.parameters()) for model in models)
    scratch = max((layer.in_features * layer.out_features * torch.empty((), dtype=dtype).element_size()
                   for model in models for layer in model.modules() if isinstance(layer, QuantizedLinear)), default=0)
    free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
    # Activations, VAE tuilé et intermédiaires de déquantification bornés.
    required = resident + scratch + 2 * 1024**3
    if required > free_bytes:
        logging.getLogger(__name__).info("Krea 2 : offload par sous-module (VRAM disponible insuffisante pour un composant entier).")
        pipeline.enable_sequential_cpu_offload(device=device)
    else:
        pipeline.enable_model_cpu_offload(device=device)


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

        for path in (config.checkpoint, config.text_encoder):
            inspect_weight_format(path, allow_quantized=True)
        inspect_weight_format(config.vae)
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
        configure_krea2_memory(pipeline, device=device, offload=offload, dtype=dtype)
        return pipeline
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            f"Chargement Krea 2 impossible ({config.checkpoint}, {config.text_encoder}, {config.vae}) : {exc}. "
            "Vérifiez les fichiers locaux et réinstallez les dépendances d'inférence verrouillées."
        ) from exc
