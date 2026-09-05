"""Chemins des modèles, caches et artefacts locaux."""

from __future__ import annotations

from dataclasses import dataclass
import os
import stat
from pathlib import Path
import tempfile
from typing import Mapping

from pulid_app.config import AppConfig
from pulid_app.exceptions import ExternalDriveNotMountedError


ANTELOPEV2_REQUIRED_FILES = frozenset(
    {
        "1k3d68.onnx",
        "2d106det.onnx",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    }
)


@dataclass(frozen=True)
class ModelInventory:
    pulid_checkpoints: tuple[Path, ...]
    antelope_dir: Path | None
    antelope_missing_files: tuple[str, ...]
    sdxl_candidates: tuple[Path, ...]
    warnings: tuple[str, ...] = ()


def external_cache_paths(models_root: Path) -> dict[str, Path]:
    """Retourne les emplacements imposés aux bibliothèques de modèles."""

    return {
        "HF_HOME": models_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": models_root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": models_root / "huggingface" / "transformers",
        "TORCH_HOME": models_root / "torch",
        "XDG_CACHE_HOME": models_root / "other",
        "MPLCONFIGDIR": models_root / "other" / "matplotlib",
    }


def configure_external_model_caches(models_root: Path) -> Mapping[str, str]:
    """Redirige les caches lourds avant tout import d'une bibliothèque ML.

    La fonction est idempotente et remplace volontairement toute valeur héritée :
    la racine de modèles configurée reste l'unique source de vérité.
    """

    configured: dict[str, str] = {}
    for name, path in external_cache_paths(models_root).items():
        value = str(path.resolve(strict=False))
        os.environ[name] = value
        configured[name] = value
    # Albumentations effectue sinon une requête PyPI à chaque nouvel environnement.
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    configured["NO_ALBUMENTATIONS_UPDATE"] = "1"
    return configured


def cache_env_violations(
    models_root: Path,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Signale les caches effectifs absents ou situés hors de ``models_root``."""

    selected = os.environ if environ is None else environ
    root = models_root.expanduser().resolve(strict=False)
    violations: list[str] = []
    for name in external_cache_paths(root):
        raw_value = selected.get(name)
        if not raw_value:
            violations.append(f"{name} n'est pas défini")
            continue
        path = Path(raw_value).expanduser().resolve(strict=False)
        try:
            path.relative_to(root)
        except ValueError:
            violations.append(f"{name} pointe hors de models_root : {path}")
    return tuple(violations)


def _unique_sorted(paths: list[Path]) -> tuple[Path, ...]:
    return tuple(sorted(set(paths), key=lambda item: str(item).casefold()))


def _technical_directory(name: str) -> bool:
    lowered = name.casefold()
    return lowered in {".git", ".venv", "venv", "__pycache__", "node_modules", "uv"} or lowered.startswith(
        ("uv-", "cpython-", "pypy-")
    )


def _inventory_tree(root: Path, warnings: list[str]) -> tuple[list[Path], list[Path]]:
    """Walk once, pruning runtimes and never following directory reparse points.

    Python 3.11 rglob can raise on broken Windows junctions. Catch errors at
    each directory/entry so a disappearing or inaccessible sibling is harmless.
    File symlinks (e.g. Hugging Face snapshots) remain valid model candidates.
    """
    files: list[Path] = []
    directories: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if _technical_directory(entry.name):
                        continue
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        is_reparse = bool(
                            getattr(metadata, "st_file_attributes", 0)
                            & stat.FILE_ATTRIBUTE_REPARSE_POINT
                        )
                        if stat.S_ISDIR(metadata.st_mode):
                            if not is_reparse:
                                directories.append(path)
                                pending.append(path)
                        elif entry.is_file():
                            files.append(path)
                    except OSError as exc:
                        warnings.append(f"Entrée ignorée : {path} ({exc})")
        except OSError as exc:
            warnings.append(f"Dossier non parcouru : {directory} ({exc})")
    return files, directories


def inspect_models(config: AppConfig) -> ModelInventory:
    """Inventorie les modèles locaux sans charger ni modifier les runtimes."""
    warnings: list[str] = []

    def is_file(path: Path) -> bool:
        try:
            return path.is_file()
        except OSError as exc:
            warnings.append(f"Fichier inaccessible : {path} ({exc})")
            return False

    files, directories = _inventory_tree(config.models_root, warnings)
    safetensors = [p for p in files if p.suffix.casefold() == ".safetensors"]
    pulid_paths = [path for path in safetensors if "pulid" in path.name.casefold()]
    if is_file(config.pulid.checkpoint):
        pulid_paths.append(config.pulid.checkpoint)
    sdxl_paths = [
        path for path in safetensors
        if "pulid" not in path.name.casefold()
        and "vae" not in path.name.casefold().replace("bakedvae", "")
    ]
    if is_file(config.sdxl.checkpoint):
        sdxl_paths.append(config.sdxl.checkpoint)

    antelope_dir = config.insightface.model_dir
    try:
        configured_antelope_exists = antelope_dir.is_dir()
    except OSError as exc:
        warnings.append(f"AntelopeV2 inaccessible : {antelope_dir} ({exc})")
        configured_antelope_exists = False
    if not configured_antelope_exists:
        matches = sorted(
            (p for p in directories if p.name == config.insightface.model_name),
            key=lambda item: str(item).casefold(),
        )
        antelope_dir = matches[0] if matches else None
    missing = tuple(
        sorted(name for name in ANTELOPEV2_REQUIRED_FILES if not is_file(antelope_dir / name))
    ) if antelope_dir is not None else ()
    return ModelInventory(
        pulid_checkpoints=_unique_sorted(pulid_paths),
        antelope_dir=antelope_dir,
        antelope_missing_files=missing,
        sdxl_candidates=_unique_sorted(sdxl_paths),
        warnings=tuple(warnings),
    )


def ensure_writable_directory(path: Path) -> None:
    """Crée le dossier si nécessaire et vérifie réellement son écriture."""

    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"Le chemin n'est pas un dossier : {path}")
    try:
        with tempfile.NamedTemporaryFile(prefix=".pulid-write-test-", dir=path):
            pass
    except OSError as exc:
        raise PermissionError(f"Le dossier n'est pas accessible en écriture : {path}") from exc


def require_models_root(models_root: Path) -> Path:
    """Exige une racine de modèles disponible et, sous /Volumes, montée."""

    root = models_root.expanduser().resolve(strict=False)
    parts = root.parts
    expected_mount = (
        Path("/Volumes") / parts[2]
        if len(parts) >= 3 and parts[1] == "Volumes"
        else None
    )
    if not root.is_dir() or (
        expected_mount is not None and not expected_mount.is_mount()
    ):
        raise ExternalDriveNotMountedError(root)
    return root


def resolve_sdxl_checkpoint(config: AppConfig, model_name: str | None) -> Path:
    """Résout un checkpoint nommé dans le dossier du modèle SDXL configuré."""

    if model_name is None:
        return config.sdxl.checkpoint

    selected = model_name.strip()
    if not selected:
        raise ValueError("L'option --model ne peut pas être vide.")
    if selected in {".", ".."} or "/" in selected or "\\" in selected:
        raise ValueError(
            "L'option --model attend uniquement un nom de modèle, pas un chemin."
        )

    lowered = selected.casefold()
    for extension in (".safetensors", ".safetensor"):
        if lowered.endswith(extension):
            selected = selected[: -len(extension)]
            break
    if not selected:
        raise ValueError("Le nom fourni à --model est invalide.")

    return (
        config.sdxl.checkpoint.parent / f"{selected}.safetensors"
    ).resolve(strict=False)
