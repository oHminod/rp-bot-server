from __future__ import annotations

from io import StringIO
import hashlib
from pathlib import Path
import sys
import zipfile

import pytest
from rich.console import Console
import yaml

import pulid_app.installer as installer
from pulid_app.config import load_config
from pulid_app.installer import (
    HuggingFaceAsset,
    InstallerError,
    choose_checkpoint,
    confirm_antelope_license,
    ensure_huggingface_asset,
    find_existing_models_root,
    install_antelope_archive,
    prompt_models_root,
    read_local_installation,
    resolve_models_root,
    select_sdxl_checkpoint,
    validate_sdxl_config_tree,
    write_local_config,
)


def _console() -> Console:
    return Console(file=StringIO(), force_terminal=False)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_resolve_models_root_accepts_parent_or_final_directory(tmp_path: Path) -> None:
    parent = tmp_path / "models-parent"
    final = tmp_path / "PuLID_models"

    assert resolve_models_root(parent, project_root=tmp_path) == parent / "PuLID_models"
    assert resolve_models_root(final, project_root=tmp_path) == final
    assert resolve_models_root(
        tmp_path / "pulid_MODELS", project_root=tmp_path
    ) == (tmp_path / "pulid_MODELS")


def test_prompt_models_root_uses_default_or_normalizes_custom_path(tmp_path: Path) -> None:
    default_answers = iter([""])
    custom_answers = iter(["n", str(tmp_path / "external")])

    assert prompt_models_root(
        project_root=tmp_path,
        input_fn=lambda _prompt: next(default_answers),
    ) == tmp_path / "PuLID_models"
    assert prompt_models_root(
        project_root=tmp_path,
        input_fn=lambda _prompt: next(custom_answers),
    ) == tmp_path / "external" / "PuLID_models"


def test_existing_installation_is_reused_from_local_config(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    models_root = tmp_path / "external" / "PuLID_models"
    checkpoint = models_root / "checkpoints" / "custom.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config_path = project_root / "config" / "local.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        "models_root: "
        + models_root.as_posix()
        + "\nsdxl:\n  checkpoint: checkpoints/custom.safetensors\n",
        encoding="utf-8",
    )

    assert read_local_installation(
        config_path=config_path,
        project_root=project_root,
    ) == (models_root, checkpoint)
    assert find_existing_models_root(
        project_root=project_root,
        config_path=config_path,
        environ={},
    ) == models_root


def test_existing_default_models_root_is_reused_without_config(tmp_path: Path) -> None:
    models_root = tmp_path / "PuLID_models"
    models_root.mkdir()

    assert find_existing_models_root(
        project_root=tmp_path,
        config_path=tmp_path / "missing.yaml",
        environ={},
    ) == models_root


def test_existing_sdxl_checkpoint_skips_all_questions(tmp_path: Path) -> None:
    models_root = tmp_path / "PuLID_models"
    first = models_root / "checkpoints" / "a.safetensors"
    preferred = models_root / "checkpoints" / "preferred.safetensors"
    first.parent.mkdir(parents=True)
    first.touch()
    preferred.touch()

    selected = select_sdxl_checkpoint(
        models_root,
        _console(),
        preferred_checkpoint=preferred,
        input_fn=lambda _prompt: pytest.fail("aucune question attendue"),
    )

    assert selected == preferred


def test_sdxl_installation_can_be_deferred(tmp_path: Path) -> None:
    console = _console()
    prompts: list[str] = []

    def answer_no(prompt: str) -> str:
        prompts.append(prompt)
        return "non"

    selected = select_sdxl_checkpoint(
        tmp_path / "PuLID_models",
        console,
        input_fn=answer_no,
    )

    assert selected is None
    assert len(prompts) == 1
    assert "ultérieurement" in prompts[0]
    assert "différée" in console.file.getvalue()


def test_sdxl_skip_mode_does_not_ask_questions(tmp_path: Path) -> None:
    selected = select_sdxl_checkpoint(
        tmp_path / "PuLID_models",
        _console(),
        mode="skip",
        input_fn=lambda _prompt: pytest.fail("aucune question attendue"),
    )

    assert selected is None


def test_huggingface_asset_is_idempotent_and_repairs_invalid_file(
    tmp_path: Path,
) -> None:
    content = b"valid model"
    asset = HuggingFaceAsset(
        name="test",
        relative_path="models/Target.bin",
        repository="owner/repo",
        filename="remote.bin",
        sha256=_digest(content),
    )
    target = tmp_path / asset.relative_path
    target.parent.mkdir(parents=True)
    target.write_bytes(content)

    ensure_huggingface_asset(
        tmp_path,
        asset,
        _console(),
        downloader=lambda **_kwargs: pytest.fail("aucun téléchargement attendu"),
    )

    target.write_bytes(b"broken")

    def downloader(**kwargs: object) -> str:
        local_dir = Path(kwargs["local_dir"])
        downloaded = local_dir / str(kwargs["filename"])
        downloaded.write_bytes(content)
        return str(downloaded)

    repaired = ensure_huggingface_asset(
        tmp_path,
        asset,
        _console(),
        downloader=downloader,
    )

    assert repaired == target
    assert target.read_bytes() == content
    assert not (target.parent / "remote.bin").exists()


def test_install_antelope_archive_replaces_only_managed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {"one.onnx": b"one", "two.onnx": b"two"}
    checksums = {name: _digest(content) for name, content in files.items()}
    monkeypatch.setattr(installer, "ANTELOPEV2_REQUIRED_FILES", frozenset(files))
    monkeypatch.setattr(installer, "ANTELOPE_FILE_SHA256", checksums)
    archive_path = tmp_path / "antelope.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in files.items():
            archive.writestr(f"antelopev2/{name}", content)
    destination = tmp_path / "antelopev2"
    destination.mkdir()
    (destination / "one.onnx").write_bytes(b"broken")
    unrelated = destination / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")

    install_antelope_archive(tmp_path, archive_path)

    assert {name: (destination / name).read_bytes() for name in files} == files
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_antelope_license_must_be_accepted_before_download(tmp_path: Path) -> None:
    console = _console()

    with pytest.raises(InstallerError, match="n'a pas été acceptée"):
        confirm_antelope_license(
            tmp_path,
            console,
            input_fn=lambda _prompt: "non",
        )

    assert "recherche non commerciale" in console.file.getvalue()
    confirm_antelope_license(
        tmp_path,
        _console(),
        accepted=True,
        input_fn=lambda _prompt: pytest.fail("aucune question attendue"),
    )


def test_existing_antelope_does_not_repeat_license_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"model"
    monkeypatch.setattr(installer, "ANTELOPEV2_REQUIRED_FILES", {"model.onnx"})
    monkeypatch.setattr(
        installer,
        "ANTELOPE_FILE_SHA256",
        {"model.onnx": _digest(content)},
    )
    model = tmp_path / "antelopev2" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(content)

    confirm_antelope_license(
        tmp_path,
        _console(),
        input_fn=lambda _prompt: pytest.fail("aucune question attendue"),
    )


def test_choose_checkpoint_uses_numbered_selection(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"

    assert choose_checkpoint(
        (first, second), input_fn=lambda _prompt: "2"
    ) == second
    with pytest.raises(InstallerError, match="Aucun checkpoint"):
        choose_checkpoint(())


def test_write_local_config_persists_root_and_checkpoint(tmp_path: Path) -> None:
    models_root = tmp_path / "chosen" / "PuLID_models"
    checkpoint = models_root / "checkpoints" / "custom.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    default_config = tmp_path / "default.yaml"
    default_config.write_text(
        """
models_root: PuLID_models
sdxl:
  checkpoint: checkpoints/default.safetensors
  config_dir: sdxl/config
pulid:
  checkpoint: pulid_v1.1.safetensors
insightface:
  model_root: .
  model_name: antelopev2
text_embedding:
  checkpoint: text_embedding/bge-m3-Q8_0.gguf
  model_id: text-embedding-bge-m3
  dimensions: 1024
  context_size: 8192
  batch_size: 8192
  threads: 0
outputs_dir: outputs
identity_cache_dir: cache/identity
device:
  preferred: mps
  dtype: float16
""",
        encoding="utf-8",
    )
    local_config = tmp_path / "local.yaml"

    write_local_config(
        models_root,
        checkpoint,
        default_config=default_config,
        destination=local_config,
    )
    raw = yaml.safe_load(local_config.read_text(encoding="utf-8"))
    loaded = load_config(local_config)

    assert raw["models_root"] == models_root.as_posix()
    assert raw["sdxl"]["checkpoint"] == "checkpoints/custom.safetensors"
    assert raw["device"]["preferred"] == (
        "mps" if sys.platform == "darwin" else "cuda" if sys.platform == "win32" else "cpu"
    )
    assert loaded.models_root == models_root
    assert loaded.sdxl.checkpoint == checkpoint


def test_write_local_config_keeps_default_checkpoint_when_sdxl_is_deferred(
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "chosen" / "PuLID_models"
    default_config = tmp_path / "default.yaml"
    default_config.write_text(
        """
models_root: PuLID_models
sdxl:
  checkpoint: checkpoints/default.safetensors
pulid:
  checkpoint: pulid_v1.1.safetensors
insightface:
  model_root: .
  model_name: antelopev2
outputs_dir: outputs
identity_cache_dir: cache/identity
device:
  preferred: cpu
  dtype: float32
""",
        encoding="utf-8",
    )
    local_config = tmp_path / "local.yaml"

    write_local_config(
        models_root,
        None,
        default_config=default_config,
        destination=local_config,
    )
    raw = yaml.safe_load(local_config.read_text(encoding="utf-8"))
    loaded = load_config(local_config)

    assert raw["sdxl"]["checkpoint"] == "checkpoints/default.safetensors"
    assert loaded.sdxl.checkpoint == (
        models_root / "checkpoints" / "default.safetensors"
    )


def test_validate_sdxl_config_tree_rejects_missing_files_and_weights(
    tmp_path: Path,
) -> None:
    with pytest.raises(InstallerError, match="fichiers manquants"):
        validate_sdxl_config_tree(tmp_path)

    for relative in installer.SDXL_REQUIRED_CONFIG_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    count, _size = validate_sdxl_config_tree(tmp_path)
    assert count == len(installer.SDXL_REQUIRED_CONFIG_FILES)

    (tmp_path / "unet" / "weights.safetensors").write_bytes(b"weights")
    with pytest.raises(InstallerError, match="poids"):
        validate_sdxl_config_tree(tmp_path)


def test_local_models_root_is_portable_after_project_move(tmp_path, monkeypatch):
    import shutil
    from pulid_app.config import load_config
    root = tmp_path / 'original'
    (root / 'config').mkdir(parents=True)
    models = root / 'models' / 'PuLID_models'
    models.mkdir(parents=True)
    default = root / 'config/default.yaml'
    default.write_text('''models_root: ignored
sdxl:
  checkpoint: checkpoints/absent.safetensors
pulid:
  checkpoint: pulid.safetensors
insightface:
  model_root: .
  model_name: antelopev2
outputs_dir: outputs
identity_cache_dir: cache/identity
''')
    destination = root / 'config/local.yaml'
    write_local_config(models, None, default_config=default, destination=destination, project_root=root)
    assert yaml.safe_load(destination.read_text())['models_root'] == 'models/PuLID_models'
    moved = tmp_path / 'moved with spaces'
    shutil.move(root, moved)
    monkeypatch.setattr('pulid_app.config.PROJECT_ROOT', moved)
    monkeypatch.delenv('PULID_MODELS_ROOT', raising=False)
    loaded = load_config(moved / 'config/local.yaml')
    assert loaded.models_root == moved / 'models/PuLID_models'
    assert not loaded.sdxl.checkpoint.exists()
