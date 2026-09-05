from __future__ import annotations

from pathlib import Path

from pulid_app.config import (
    AppConfig,
    DeviceConfig,
    InsightFaceConfig,
    PuLIDConfig,
    SDXLConfig,
)
from pulid_app.paths import (
    ANTELOPEV2_REQUIRED_FILES,
    cache_env_violations,
    configure_external_model_caches,
    ensure_writable_directory,
    inspect_models,
)


def _config(tmp_path: Path) -> AppConfig:
    models = tmp_path / "models"
    return AppConfig(
        models_root=models,
        sdxl=SDXLConfig(models / "checkpoints" / "realvisxl.safetensors"),
        pulid=PuLIDConfig(models / "pulid_v1.1.safetensors"),
        insightface=InsightFaceConfig(models, "antelopev2"),
        outputs_dir=tmp_path / "outputs",
        identity_cache_dir=tmp_path / "cache" / "identity",
        device=DeviceConfig(),
        source_path=tmp_path / "config.yaml",
    )


def test_inspect_models_finds_expected_inventory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.models_root.mkdir()
    config.sdxl.checkpoint.parent.mkdir()
    config.sdxl.checkpoint.touch()
    config.pulid.checkpoint.touch()
    config.insightface.model_dir.mkdir()
    for name in ANTELOPEV2_REQUIRED_FILES:
        (config.insightface.model_dir / name).touch()

    inventory = inspect_models(config)

    assert inventory.sdxl_candidates == (config.sdxl.checkpoint,)
    assert inventory.pulid_checkpoints == (config.pulid.checkpoint,)
    assert inventory.antelope_dir == config.insightface.model_dir
    assert inventory.antelope_missing_files == ()


def test_inspect_models_reports_missing_antelope_file(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.insightface.model_dir.mkdir(parents=True)
    missing = sorted(ANTELOPEV2_REQUIRED_FILES)[0]
    for name in ANTELOPEV2_REQUIRED_FILES - {missing}:
        (config.insightface.model_dir / name).touch()

    inventory = inspect_models(config)

    assert inventory.antelope_missing_files == (missing,)


def test_configure_external_model_caches(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HF_HOME", "/an/internal/cache")

    configured = configure_external_model_caches(tmp_path)

    assert configured["HF_HOME"] == str(tmp_path / "huggingface")
    assert configured["TORCH_HOME"] == str(tmp_path / "torch")
    assert configured["MPLCONFIGDIR"] == str(tmp_path / "other" / "matplotlib")
    assert configured["NO_ALBUMENTATIONS_UPDATE"] == "1"


def test_ensure_writable_directory_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "new" / "outputs"

    ensure_writable_directory(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_cache_env_violations_reports_missing_and_outside_paths(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    configured = configure_external_model_caches(models_root)
    configured.pop("TORCH_HOME")
    configured["HF_HOME"] = str(tmp_path / "internal")

    violations = cache_env_violations(models_root, configured)

    assert any("TORCH_HOME n'est pas défini" in item for item in violations)
    assert any("HF_HOME pointe hors" in item for item in violations)


def test_configured_cache_env_is_accepted(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    configured = configure_external_model_caches(models_root)

    assert cache_env_violations(models_root, configured) == ()


def test_inventory_prunes_uv_runtimes_before_scanning(tmp_path, monkeypatch):
    import os
    config = _config(tmp_path)
    technical = config.models_root / 'other' / 'uv-python-windows'
    broken = technical / 'cpython-3.11-windows-x86_64-none'
    broken.mkdir(parents=True)
    config.sdxl.checkpoint.parent.mkdir()
    config.sdxl.checkpoint.touch()
    original = os.scandir

    def guarded_scandir(path):
        if Path(path).is_relative_to(technical):
            raise AssertionError('uv/Python must not be traversed')
        return original(path)

    monkeypatch.setattr(os, 'scandir', guarded_scandir)
    inventory = inspect_models(config)
    assert inventory.sdxl_candidates == (config.sdxl.checkpoint,)
    assert not inventory.warnings


def test_inventory_reports_disappearing_directory_and_keeps_siblings(tmp_path, monkeypatch):
    import os
    config = _config(tmp_path)
    lost = config.models_root / 'disappearing'
    lost.mkdir(parents=True)
    config.sdxl.checkpoint.parent.mkdir()
    config.sdxl.checkpoint.touch()
    original = os.scandir

    def broken_scandir(path):
        if Path(path) == lost:
            raise FileNotFoundError(3, 'WinError 3: path not found', str(path))
        return original(path)

    monkeypatch.setattr(os, 'scandir', broken_scandir)
    inventory = inspect_models(config)
    assert inventory.sdxl_candidates == (config.sdxl.checkpoint,)
    assert str(lost) in inventory.warnings[0]


def test_inventory_skips_directory_symlinks_but_accepts_model_file_symlinks(tmp_path):
    import pytest
    config = _config(tmp_path)
    config.models_root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'unexpected.safetensors').touch()
    target = outside / 'linked.safetensors'
    target.touch()
    link = config.models_root / 'linked.safetensors'
    try:
        (config.models_root / 'cycle').symlink_to(config.models_root, target_is_directory=True)
        (config.models_root / 'outside').symlink_to(outside, target_is_directory=True)
        (config.models_root / 'broken').symlink_to(tmp_path / 'absent', target_is_directory=True)
        link.symlink_to(target)
    except OSError:
        pytest.skip('Symlink creation requires permission on this host')
    assert inspect_models(config).sdxl_candidates == (link,)


def test_inventory_prunes_windows_junction_attributes(tmp_path, monkeypatch):
    import os
    import stat
    from types import SimpleNamespace
    config = _config(tmp_path)
    config.models_root.mkdir()
    junction = config.models_root / 'junction'
    junction.mkdir()
    (junction / 'hidden.safetensors').touch()
    original = os.scandir

    class FakeScan:
        def __enter__(self):
            return iter([SimpleNamespace(
                name='junction', path=str(junction),
                stat=lambda **kwargs: SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT),
            )])
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(os, 'scandir', lambda path: FakeScan() if Path(path) == config.models_root else original(path))
    assert inspect_models(config).sdxl_candidates == ()
