"""Explicit model paths must survive every installer layer without a prompt."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
from rich.console import Console
import yaml

import pulid_app.installer as installer


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("via_cli", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_python_installation_preserves_explicit_root(tmp_path, monkeypatch, via_cli, existing):
    selected = tmp_path / "Mes modeles"
    if existing:
        selected.mkdir()
    fallback = tmp_path / "PuLID_models"
    fallback.mkdir()
    monkeypatch.setenv("PULID_MODELS_ROOT", str(fallback if via_cli else selected))
    monkeypatch.setattr(installer, "read_local_installation", lambda: (fallback, None))
    monkeypatch.setattr(installer, "prompt_models_root", lambda: pytest.fail("Unexpected prompt"))
    seen = []
    monkeypatch.setattr(installer, "configure_external_model_caches", lambda root: seen.append(root))
    monkeypatch.setattr(installer, "confirm_antelope_license", lambda root, *a, **kw: seen.append(root) or True)
    monkeypatch.setattr(installer, "select_sdxl_checkpoint", lambda root, *a, **kw: seen.append(root))
    monkeypatch.setattr(installer, "prepare_required_assets", lambda root, *a, **kw: seen.append(root))
    write_config = installer.write_local_config
    config = tmp_path / "config/local.yaml"
    monkeypatch.setattr(installer, "write_local_config", lambda root, checkpoint: write_config(
        root, checkpoint, destination=config, project_root=tmp_path,
    ))
    args = installer.build_parser().parse_args([
        "--sdxl", "skip", "--accept-insightface-license",
        *(["--models-root", str(selected)] if via_cli else []),
    ])
    assert installer.run_installation(args, Console(quiet=True)) == 0
    assert seen == [selected.resolve()] * 4
    assert (selected / "checkpoints").is_dir()
    assert not (selected / "PuLID_models").exists()
    assert yaml.safe_load(config.read_text())["models_root"] == "Mes modeles"


def test_relative_explicit_root_and_environment_detection(tmp_path):
    selected = tmp_path / "Mes modeles"
    selected.mkdir()
    assert installer.resolve_explicit_models_root("Mes modeles", project_root=tmp_path) == selected
    assert installer.find_existing_models_root(
        project_root=tmp_path, config_path=tmp_path / "missing.yaml",
        environ={"PULID_MODELS_ROOT": "Mes modeles"},
    ) == selected


@pytest.mark.parametrize("platform", ["macos", "windows"])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_native_bootstrap_selects_exact_root_without_stdin(tmp_path, platform, existing, relative):
    if platform == "windows" and sys.platform != "win32":
        pytest.skip("Requires native cmd.exe")
    if platform == "macos" and sys.platform != "darwin":
        pytest.skip("Requires macOS Bash")
    project = tmp_path / "project"
    project.mkdir()
    selected = project / "Mes modeles"
    if existing:
        selected.mkdir()
    # A default directory must never override an explicit non-existent root.
    (project / "PuLID_models").mkdir()
    environment = {**os.environ, "PULID_MODELS_ROOT": "Mes modeles" if relative else str(selected)}
    if platform == "macos":
        source = (ROOT / "install_macos.sh").read_text()
        # Run the real path selection and mkdir, stopping before uv/model setup.
        prefix = source.split("export PULID_MODELS_ROOT", 1)[0]
        script = project / "selection.sh"
        script.write_text(prefix + '\nprintf "SELECTED=%s\\n" "${PULID_MODELS_ROOT}"\n')
        command = ["/bin/bash", str(script), "--production"]
    else:
        source = (ROOT / "install_windows.bat").read_text()
        prefix = source.split("\n:models_root_available\n", 1)[0] + "\n"
        script = project / "selection.bat"
        script.write_bytes(("@chcp 65001 >nul\n" + prefix +
            ':models_root_available\necho SELECTED=%PULID_MODELS_ROOT%\nexit /b 0\n'
            ':error_exit\nexit /b 1\n').replace("\n", "\r\n").encode())
        command = [os.environ["COMSPEC"], "/d", "/c", str(script), "--production"]
    result = subprocess.run(command, cwd=tmp_path, env=environment, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = next(line.removeprefix("SELECTED=") for line in result.stdout.splitlines() if line.startswith("SELECTED="))
    assert Path(actual).resolve() == selected.resolve()
    assert selected.is_dir()
    assert not (selected / "PuLID_models").exists()
