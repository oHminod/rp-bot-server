from pathlib import Path

import pytest

from pulid_app.config import Krea2Config
from pulid_app.exceptions import ModelNotFoundError
from pulid_app.models.krea2_catalog import krea2_catalog, select_krea2_models


@pytest.fixture
def config(tmp_path):
    return Krea2Config(checkpoint=tmp_path / "custom/krea/default.safetensors",
                       text_encoder=tmp_path / "custom/qwen/default.safetensors",
                       text_encoder_config_dir=tmp_path / "custom/qwen/config",
                       vae=tmp_path / "vae.safetensors")


def write_model(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake weights")
    return path


def test_catalog_is_empty_without_installation_and_never_creates_directories(config, tmp_path):
    assert krea2_catalog(config) == {"models": [], "text_encoders": []}
    with pytest.raises(ModelNotFoundError, match="manuellement"):
        select_krea2_models(config)
    assert list(tmp_path.iterdir()) == []


def test_catalog_uses_arbitrary_names_and_configured_directories(config):
    for name in ("z-custom.safetensors", "A FP8.SAFETENSORS", "éclair.safetensors", "notes.txt"):
        write_model(config.checkpoint.parent / name)
    write_model(config.checkpoint.parent / "nested/hidden.safetensors")
    (config.checkpoint.parent / "folder.safetensors").mkdir()
    encoder = write_model(config.text_encoder.parent / "Qwen personnel.safetensors")
    write_model(config.text_encoder_config_dir / "config.json")
    catalog = krea2_catalog(config)
    assert [item["name"] for item in catalog["models"]] == [
        "A FP8.SAFETENSORS", "z-custom.safetensors", "éclair.safetensors"]
    assert catalog["models"][0]["default"] is True
    assert catalog["text_encoders"] == [{"name": encoder.name, "filename": encoder.name, "default": True}]
    selected = select_krea2_models(config)
    assert selected.checkpoint.name == "A FP8.SAFETENSORS"
    assert selected.text_encoder == encoder
    assert selected.text_encoder_config_dir == config.text_encoder_config_dir
    assert selected.vae == config.vae
    write_model(config.checkpoint)
    assert select_krea2_models(config).checkpoint == config.checkpoint
    assert [item["name"] for item in krea2_catalog(config)["models"] if item["default"]] == [config.checkpoint.name]


@pytest.mark.parametrize("field", ["model", "text_encoder"])
@pytest.mark.parametrize("name", ["../escape.safetensors", r"..\escape.safetensors", "/tmp/a.safetensors",
                                  r"C:\models\a.safetensors", "", "bad\n.safetensors", "missing.safetensors"])
def test_selection_rejects_paths_and_unknown_filenames(config, field, name):
    write_model(config.checkpoint)
    write_model(config.text_encoder)
    with pytest.raises((ValueError, ModelNotFoundError)):
        select_krea2_models(config, **{field: name})


def test_catalog_is_refreshed_and_rejects_external_symlinks(config, tmp_path):
    external = write_model(tmp_path / "external.safetensors")
    config.checkpoint.parent.mkdir(parents=True)
    config.checkpoint.symlink_to(external)
    assert krea2_catalog(config)["models"] == []
    with pytest.raises(ModelNotFoundError):
        select_krea2_models(config)
    with pytest.raises(ModelNotFoundError):
        select_krea2_models(config, model=config.checkpoint.name)
    local = write_model(config.checkpoint.parent / "local.safetensors")
    write_model(config.text_encoder)
    assert select_krea2_models(config).checkpoint == local
    local.unlink()
    assert krea2_catalog(config)["models"] == []
