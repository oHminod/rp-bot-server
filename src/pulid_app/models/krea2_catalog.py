"""Découverte locale des poids Krea/Qwen, sans lire ni charger les tenseurs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pulid_app.config import Krea2Config
from pulid_app.exceptions import ModelNotFoundError


def checkpoint_files(configured: Path) -> list[Path]:
    """Accepte tout nom Safetensors, limité au dossier du fichier configuré."""
    directory = configured.parent.resolve()
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir()
         if path.suffix.casefold() == ".safetensors" and path.is_file()
         and path.resolve().is_relative_to(directory)),
        key=lambda path: (path.name.casefold(), path.name),
    )


def _default_file(configured: Path, files: list[Path]) -> Path:
    return next((path for path in files if path.name == configured.name),
                files[0] if files else configured)


def _options(configured: Path) -> list[dict[str, str | bool]]:
    files = checkpoint_files(configured)
    default = _default_file(configured, files)
    return [{"name": path.name, "filename": path.name, "default": path == default}
            for path in files]


def krea2_catalog(config: Krea2Config) -> dict[str, list[dict[str, str | bool]]]:
    return {"models": _options(config.checkpoint),
            "text_encoders": _options(config.text_encoder)}


def _select_file(configured: Path, selected: str | None, field: str) -> Path:
    if selected is not None and (
        not selected or selected in {".", ".."}
        or any(character in selected for character in ("/", "\\"))
        or any(ord(character) < 32 or ord(character) == 127 for character in selected)
    ):
        raise ValueError(f"{field} attend un nom de fichier de GET /models/krea2, pas un chemin.")
    files = checkpoint_files(configured)
    if selected is None:
        if not files:
            raise ModelNotFoundError(
                f"Aucun poids {field} dans {configured.parent}. "
                "Ajoutez manuellement un fichier .safetensors compatible puis actualisez GET /models/krea2."
            )
        return _default_file(configured, files)
    for path in files:
        if path.name == selected:
            return path
    raise ModelNotFoundError(
        f"{field} introuvable : {selected} dans {configured.parent}. "
        "Choisissez un fichier renvoyé par GET /models/krea2."
    )


def select_krea2_models(
    config: Krea2Config, *, model: str | None = None, text_encoder: str | None = None,
) -> Krea2Config:
    return replace(config,
                   checkpoint=_select_file(config.checkpoint, model, "model"),
                   text_encoder=_select_file(config.text_encoder, text_encoder, "text_encoder"))
