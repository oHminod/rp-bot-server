from dataclasses import replace
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console

import pulid_app.installer as installer
from pulid_app.config import load_config
from pulid_app.exceptions import ModelLoadError, ModelNotFoundError
from pulid_app.models.krea2 import validate_krea2_assets
from test_server import _write_config


@pytest.fixture
def downloads(monkeypatch):
    contents = {asset.filename: f'{{"fixture": "{asset.filename}"}}'.encode()
                for asset in installer.QWEN3VL_CONFIG_ASSETS}
    monkeypatch.setattr(installer, "QWEN3VL_CONFIG_ASSETS", tuple(
        replace(asset, sha256=hashlib.sha256(contents[asset.filename]).hexdigest())
        for asset in installer.QWEN3VL_CONFIG_ASSETS))
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["repo_id"] == "Qwen/Qwen3-VL-4B-Instruct"
        assert kwargs["revision"] == installer.QWEN3VL_CONFIG_REVISION
        assert kwargs["filename"] in contents  # Aucun poids, index ou snapshot global.
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(contents[kwargs["filename"]])
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    return calls, contents


def test_configuration_downloads_only_shared_files_and_repairs_only_missing_or_corrupt(tmp_path, downloads):
    calls, contents = downloads
    root = tmp_path / "models"
    destination = root / "text_encoders/qwen3vl/config"
    destination.parent.mkdir(parents=True)
    weights = destination.parent / "custom FP8.safetensors"
    weights.write_bytes(b"manual weights")
    console = Console(quiet=True)
    installer.ensure_qwen3vl_configuration(root, destination, console)
    assert {call["filename"] for call in calls} == set(contents)
    assert all(call["cache_dir"] == root / "huggingface/hub" for call in calls)
    before = {name: (destination / name).stat().st_mtime_ns for name in contents}
    installer.ensure_qwen3vl_configuration(root, destination, console)
    assert len(calls) == len(contents)  # Complet : fonctionne hors ligne, aucune requête.
    assert {name: (destination / name).stat().st_mtime_ns for name in contents} == before
    (destination / "tokenizer.json").unlink()
    (destination / "config.json").write_bytes(b"broken")
    installer.ensure_qwen3vl_configuration(root, destination, console)
    assert {call["filename"] for call in calls[len(contents):]} == {"config.json", "tokenizer.json"}
    for name, content in contents.items():
        assert (destination / name).read_bytes() == content
        if name not in {"config.json", "tokenizer.json"}:
            assert (destination / name).stat().st_mtime_ns == before[name]
    assert weights.read_bytes() == b"manual weights"
    assert not (root / "krea2").exists()
    assert not (root / "vae").exists()


def test_config_only_honors_configured_directory_without_changing_yaml_or_installing_weights(tmp_path, monkeypatch, downloads):
    config_path, root = _write_config(tmp_path)
    monkeypatch.setenv("PULID_CONFIG", str(config_path))
    monkeypatch.setenv("PULID_KREA2_TEXT_ENCODER_CONFIG_DIR", "qwen/shared config")
    before = config_path.read_bytes()
    for function in ("write_local_config", "write_krea2_config", "confirm_antelope_license", "prepare_required_assets", "prepare_krea2_assets"):
        monkeypatch.setattr(installer, function, lambda *a, **k: pytest.fail("Installation de poids ou écriture YAML inattendue"))
    args = installer.build_parser().parse_args(["--models-root", str(root), "--qwen3vl-config-only"])
    assert installer.run_installation(args, Console(quiet=True)) == 0
    assert config_path.read_bytes() == before
    assert (root / "qwen/shared config/tokenizer.json").is_file()
    assert not (root / "text_encoders").exists()
    assert not (root / "vae").exists()


@pytest.mark.parametrize("failure", ["network", "checksum"])
def test_configuration_failure_reports_target_and_allows_retry(tmp_path, downloads, failure):
    destination = tmp_path / "qwen/config"
    def broken(**kwargs):
        if failure == "network":
            raise OSError("offline")
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.write_bytes(b"truncated")
        return str(target)
    with pytest.raises(installer.InstallerError) as error:
        installer.ensure_qwen3vl_configuration(tmp_path, destination, Console(quiet=True), downloader=broken)
    assert str(destination / "config.json") in str(error.value)
    installer.ensure_qwen3vl_configuration(tmp_path, destination, Console(quiet=True))
    assert (destination / "config.json").read_bytes() == downloads[1]["config.json"]


@pytest.mark.parametrize("kind", ["outside", "symlink", "directory"])
def test_configuration_rejects_unsafe_targets_before_downloading(tmp_path, downloads, kind):
    root = tmp_path / "models"
    destination = root / "qwen/config"
    external = tmp_path / "external.json"
    external.write_bytes(b"untouched")
    if kind == "outside":
        destination = tmp_path / "outside"
    else:
        destination.mkdir(parents=True)
        if kind == "symlink":
            (destination / "config.json").symlink_to(external)
        else:
            (destination / "config.json").mkdir()
    with pytest.raises(installer.InstallerError):
        installer.ensure_qwen3vl_configuration(root, destination, Console(quiet=True))
    assert not downloads[0]
    assert external.read_bytes() == b"untouched"


@pytest.mark.parametrize("missing", ["directory", "tokenizer.json", "empty"])
def test_runtime_missing_configuration_points_to_automatic_preparation(tmp_path, missing):
    path, _ = _write_config(tmp_path)
    config = load_config(path).krea2
    for weight in (config.checkpoint, config.text_encoder, config.vae):
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"fake weights")
    if missing != "directory":
        config.text_encoder_config_dir.mkdir()
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (config.text_encoder_config_dir / name).write_text("{}")
        if missing == "empty":
            (config.text_encoder_config_dir / "tokenizer.json").write_bytes(b"")
        else:
            (config.text_encoder_config_dir / missing).unlink()
    with pytest.raises((ModelLoadError, ModelNotFoundError), match="--qwen3vl-config-only") as error:
        validate_krea2_assets(config)
    assert str(config.text_encoder_config_dir) in str(error.value)
