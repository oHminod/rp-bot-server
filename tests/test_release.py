from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import tarfile
import tomllib

from scripts.build_release import MODEL_SUFFIXES, ROOT_FILES, STATIC_FILES, SCRIPT_NAMES, TREE_DIRECTORIES, build_release
from scripts.lock_environment import write_lock_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _release_source(tmp_path: Path) -> Path:
    """Use a tiny wheel for archive tests; validate the real checkout separately."""
    root = tmp_path / "source"
    for name in (*ROOT_FILES, *STATIC_FILES, *(f"scripts/{name}" for name in SCRIPT_NAMES)):
        source = PROJECT_ROOT / name
        if source.is_file():
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for directory in TREE_DIRECTORIES:
        shutil.copytree(PROJECT_ROOT / directory, root / directory,
                        ignore=shutil.ignore_patterns("*.whl", "__pycache__"), dirs_exist_ok=True)
    manifest_path = root / "runtime/wheels/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    old_hash = manifest["sha256"]
    wheel = manifest_path.parent / manifest["filename"]
    wheel.write_bytes(b"tiny release packaging fixture, never installed")
    manifest["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    lock = root / "uv.lock"
    lock.write_text(lock.read_text().replace(old_hash, manifest["sha256"]))
    write_lock_manifest(root)
    return root


def test_checkout_contains_the_locked_metal_wheel() -> None:
    manifest = json.loads((PROJECT_ROOT / "runtime/wheels/manifest.json").read_text(encoding="utf-8"))
    wheel = PROJECT_ROOT / "runtime/wheels" / manifest["filename"]
    assert wheel.is_file(), f"Wheel Metal manquante dans les sources : {wheel}"
    # A missing binary or Git LFS pointer must fail on a fresh checkout too.
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == manifest["sha256"]


def test_release_archive_is_deterministic_installable_source_without_local_data(
    tmp_path: Path,
) -> None:
    source = _release_source(tmp_path)
    first_archive, first_checksums = build_release(source, tmp_path / "first")
    second_archive, second_checksums = build_release(source, tmp_path / "second")

    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert first_checksums.read_text(encoding="ascii") == second_checksums.read_text(
        encoding="ascii"
    )

    expected_digest = hashlib.sha256(first_archive.read_bytes()).hexdigest()
    assert first_checksums.read_text(encoding="ascii") == (
        f"{expected_digest}  {first_archive.name}\n"
    )

    pyproject = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    version = pyproject["project"]["version"]
    prefix = f"pulid-{version}"
    assert first_archive.name == f"{prefix}.tar.gz"

    with tarfile.open(first_archive, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        metadata = json.load(archive.extractfile(f"{prefix}/release-metadata.json"))

    assert metadata == {
        "apiContractVersion": "1.0.0",
        "archiveFormatVersion": 1,
        "component": "pulid",
        "version": version,
    }
    assert f"{prefix}/pyproject.toml" in names
    assert f"{prefix}/uv.lock" in names
    assert f"{prefix}/.uv-version" in names
    assert f"{prefix}/scripts/check_environment.py" in names
    assert f"{prefix}/scripts/prepare_runtime_macos.sh" in names
    assert f"{prefix}/scripts/prepare_runtime_windows.ps1" in names
    assert f"{prefix}/scripts/bootstrap_windows.ps1" in names
    assert f"{prefix}/runtime/wheels/manifest.json" in names
    assert any(name.endswith("macosx_11_0_arm64.whl") for name in names)
    assert f"{prefix}/RELEASE.md" in names
    assert f"{prefix}/install_macos.sh" in names
    assert f"{prefix}/install_windows.bat" in names
    assert f"{prefix}/install_production_macos.sh" in names
    assert f"{prefix}/install_production_windows.bat" in names
    assert not any("/.git/" in name or "/.venv/" in name for name in names)
    assert not any("/tests/" in name or "/scripts/test_" in name for name in names)
    assert not any(name.endswith("config/local.yaml") for name in names)
    assert not any(name.endswith("inputs/noemie.webp") for name in names)
    assert not any(name.endswith(".DS_Store") for name in names)
    assert not any(Path(name).suffix.casefold() in MODEL_SUFFIXES for name in names)
    assert all(
        member.uid == 0 and member.gid == 0 and member.mtime == 0
        for member in members
    )


def test_release_rejects_missing_precompiled_wheel(tmp_path):
    import pytest
    source = _release_source(tmp_path)
    for wheel in (source / "runtime/wheels").glob("*.whl"):
        wheel.unlink()
    with pytest.raises(RuntimeError, match="Wheel Metal absente"):
        build_release(source, tmp_path / "dist")
