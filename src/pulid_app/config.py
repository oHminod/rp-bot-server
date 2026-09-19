"""Chargement et validation de la configuration de l'application."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


def resolve_project_root(
    *,
    module_path: Path = Path(__file__),
    prefix: Path = Path(sys.prefix),
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Retrouve la racine source en mode éditable comme en installation gérée."""

    environment = os.environ if environ is None else environ
    configured = environment.get("PULID_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve(strict=False)

    source_candidate = module_path.resolve(strict=False).parents[2]
    if (source_candidate / "config" / "default.yaml").is_file():
        return source_candidate

    managed_candidate = prefix.expanduser().resolve(strict=False).parent
    if (managed_candidate / "config" / "default.yaml").is_file():
        return managed_candidate
    return source_candidate


PROJECT_ROOT = resolve_project_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "config" / "local.yaml"
DEFAULT_PULID_REVISION = "1aa2fc7df4bf51080df39f355f9abdc1cbfefbaa"


class ConfigError(ValueError):
    """Configuration absente, invalide ou incohérente."""


@dataclass(frozen=True)
class SDXLConfig:
    checkpoint: Path
    config_dir: Path | None = None


KREA2_CHECKPOINT = "krea2/checkpoints/krea2.safetensors"
QWEN3VL_CHECKPOINT = "text_encoders/qwen3vl/qwen3vl_4b_bf16.safetensors"
QWEN3VL_CONFIG_DIR = "text_encoders/qwen3vl/config"
QWEN_IMAGE_VAE = "vae/qwen_image_vae.safetensors"


@dataclass(frozen=True)
class Krea2Config:
    checkpoint: Path
    text_encoder: Path
    text_encoder_config_dir: Path
    vae: Path


@dataclass(frozen=True)
class PuLIDConfig:
    checkpoint: Path
    source_dir: Path | None = None
    revision: str = DEFAULT_PULID_REVISION
    eva_clip_model: str = "EVA02-CLIP-L-14-336"
    eva_clip_pretrained: str = "eva_clip"
    facexlib_root: Path | None = None


@dataclass(frozen=True)
class InsightFaceConfig:
    model_root: Path
    model_name: str

    @property
    def model_dir(self) -> Path:
        return self.model_root / self.model_name


@dataclass(frozen=True)
class DeviceConfig:
    preferred: str = "mps"
    dtype: str = "float16"
    offload_strategy: str = "none"


@dataclass(frozen=True)
class TextEmbeddingConfig:
    checkpoint: Path
    model_id: str = "text-embedding-bge-m3"
    dimensions: int = 1024
    context_size: int = 8192
    batch_size: int = 8192
    # 0 laisse llama-cpp-python choisir selon les CPU logiques disponibles.
    threads: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < self.context_size:
            raise ConfigError(
                "La clé 'text_embedding.batch_size' doit être supérieure ou égale "
                "à 'text_embedding.context_size' pour un modèle d'embedding "
                "encodeur. Sinon llama.cpp interrompt nativement le processus sur "
                "les séquences longues."
            )


@dataclass(frozen=True)
class AppConfig:
    models_root: Path
    sdxl: SDXLConfig
    pulid: PuLIDConfig
    insightface: InsightFaceConfig
    outputs_dir: Path
    identity_cache_dir: Path
    device: DeviceConfig
    source_path: Path
    text_embedding: TextEmbeddingConfig | None = None
    krea2: Krea2Config | None = None


def _require_mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"La section '{key}' doit être un objet YAML.")
    return value


def _require_text(mapping: Mapping[str, Any], key: str, context: str = "") -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        qualified = f"{context}.{key}" if context else key
        raise ConfigError(f"La clé '{qualified}' doit contenir une chaîne non vide.")
    return value.strip()


def _absolute(path: str | Path, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _model_path(path: str, models_root: Path) -> Path:
    return _absolute(path, models_root)


def _bounded_integer(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = mapping.get(key, default)
    qualified = f"{context}.{key}"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"La clé '{qualified}' doit contenir un entier.")
    if not minimum <= value <= maximum:
        raise ConfigError(
            f"La clé '{qualified}' doit être comprise entre {minimum} et {maximum}."
        )
    return value


def load_config(path: str | Path | None = None, *, models_root_override: Path | None = None) -> AppConfig:
    """Charge la configuration et résout tous les chemins de façon déterministe."""

    selected = (
        path
        or os.environ.get("PULID_CONFIG")
        or (LOCAL_CONFIG_PATH if LOCAL_CONFIG_PATH.is_file() else DEFAULT_CONFIG_PATH)
    )
    source_path = _absolute(selected, PROJECT_ROOT)
    if not source_path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {source_path}")

    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide dans {source_path} : {exc}") from exc

    root = _require_mapping(raw, "racine")
    raw_models_root = models_root_override or os.environ.get("PULID_MODELS_ROOT") or _require_text(
        root, "models_root"
    )
    models_root = _absolute(raw_models_root, PROJECT_ROOT)

    sdxl = _require_mapping(root.get("sdxl"), "sdxl")
    pulid = _require_mapping(root.get("pulid"), "pulid")
    insightface = _require_mapping(root.get("insightface"), "insightface")
    device = _require_mapping(root.get("device", {}), "device")
    krea2 = _require_mapping(root.get("krea2", {}), "krea2")

    def krea_path(key: str, default: str) -> Path:
        value = os.environ.get(f"PULID_KREA2_{key.upper()}")
        if value is None:
            value = _require_text({key: krea2.get(key, default)}, key, "krea2")
        if not value.strip():
            raise ConfigError(f"Le chemin krea2.{key} doit être non vide.")
        resolved = _model_path(value, models_root)
        if not resolved.is_relative_to(models_root):
            raise ConfigError(f"krea2.{key} doit rester sous {models_root} : {resolved}")
        return resolved
    raw_text_embedding = root.get("text_embedding")
    text_embedding = (
        None
        if raw_text_embedding is None
        else _require_mapping(raw_text_embedding, "text_embedding")
    )

    outputs_dir = _absolute(_require_text(root, "outputs_dir"), PROJECT_ROOT)
    identity_cache_dir = _absolute(
        _require_text(root, "identity_cache_dir"), PROJECT_ROOT
    )
    insightface_root = _model_path(
        _require_text(insightface, "model_root", "insightface"), models_root
    )

    return AppConfig(
        models_root=models_root,
        krea2=Krea2Config(
            checkpoint=krea_path("checkpoint", KREA2_CHECKPOINT),
            text_encoder=krea_path("text_encoder", QWEN3VL_CHECKPOINT),
            text_encoder_config_dir=krea_path("text_encoder_config_dir", QWEN3VL_CONFIG_DIR),
            vae=krea_path("vae", QWEN_IMAGE_VAE),
        ),
        sdxl=SDXLConfig(
            checkpoint=_model_path(
                _require_text(sdxl, "checkpoint", "sdxl"), models_root
            ),
            config_dir=(
                _model_path(str(sdxl["config_dir"]), models_root)
                if isinstance(sdxl.get("config_dir"), str)
                and str(sdxl["config_dir"]).strip()
                else None
            ),
        ),
        pulid=PuLIDConfig(
            checkpoint=_model_path(
                _require_text(pulid, "checkpoint", "pulid"), models_root
            ),
            source_dir=_model_path(
                str(pulid.get("source_dir", "sources/PuLID")), models_root
            ),
            revision=str(
                os.environ.get("PULID_OFFICIAL_REVISION")
                or pulid.get("revision", DEFAULT_PULID_REVISION)
            ).strip(),
            eva_clip_model=str(
                pulid.get("eva_clip_model", "EVA02-CLIP-L-14-336")
            ).strip(),
            eva_clip_pretrained=str(
                pulid.get("eva_clip_pretrained", "eva_clip")
            ).strip(),
            facexlib_root=_model_path(
                str(pulid.get("facexlib_root", "facexlib/weights")), models_root
            ),
        ),
        insightface=InsightFaceConfig(
            model_root=insightface_root,
            model_name=_require_text(insightface, "model_name", "insightface"),
        ),
        outputs_dir=outputs_dir,
        identity_cache_dir=identity_cache_dir,
        device=DeviceConfig(
            preferred=str(device.get("preferred", "mps")),
            dtype=str(device.get("dtype", "float16")),
            offload_strategy=str(device.get("offload_strategy", "none")),
        ),
        source_path=source_path,
        text_embedding=(
            TextEmbeddingConfig(
                checkpoint=_model_path(
                    _require_text(text_embedding, "checkpoint", "text_embedding"),
                    models_root,
                ),
                model_id=_require_text(text_embedding, "model_id", "text_embedding"),
                dimensions=_bounded_integer(
                    text_embedding,
                    "dimensions",
                    context="text_embedding",
                    default=1024,
                    minimum=1,
                    maximum=32768,
                ),
                context_size=_bounded_integer(
                    text_embedding,
                    "context_size",
                    context="text_embedding",
                    default=8192,
                    minimum=128,
                    maximum=131072,
                ),
                batch_size=_bounded_integer(
                    text_embedding,
                    "batch_size",
                    context="text_embedding",
                    default=8192,
                    minimum=1,
                    maximum=131072,
                ),
                threads=_bounded_integer(
                    text_embedding,
                    "threads",
                    context="text_embedding",
                    default=0,
                    minimum=0,
                    maximum=1024,
                ),
            )
            if text_embedding is not None
            else None
        ),
    )
