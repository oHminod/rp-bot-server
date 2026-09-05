#!/usr/bin/env python3
"""Recreate PuLID's locked environment using only the managed bootstrap Python."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import stat
import tempfile
import urllib.request
import zipfile

CPU_WHEEL_URL = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/"
    "llama_cpp_python-0.3.35-py3-none-win_amd64.whl"
)
CPU_WHEEL_SHA256 = "31590ea000d5aff6f05f1e428048e72318a83709288159a5bd4dabec530080bb"
CPU_DLL_SHA256 = "cd91f4ed375998da4da57fedaab1b0638fba8b2af88e74a2632bc046e7fa4850"


def managed_python(root: Path, models_root: Path) -> Path:
    version = (root / ".python-version").read_text().strip()
    if sys.platform == "win32" and platform.machine().lower() in {"amd64", "x86_64"}:
        return models_root / "other" / "uv-python-windows" / f"cpython-{version}-windows-x86_64-none" / "python.exe"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return models_root / "other" / "uv-python-macos" / f"cpython-{version}-macos-aarch64-none" / "bin" / "python3.11"
    raise RuntimeError("Installation gérée : Windows x64 ou macOS Apple Silicon requis.")


def verify_bootstrap(root: Path, models_root: Path) -> Path:
    expected = managed_python(root, models_root)
    version = (root / ".python-version").read_text().strip()
    if platform.python_version() != version or Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(f"Python géré {version} requis : {expected}; reçu : {sys.executable}")
    return expected


def clean_environment(root: Path, models_root: Path) -> dict[str, str]:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"):
        environment.pop(name, None)
    suffix = "windows" if sys.platform == "win32" else "macos"
    environment.update(
        UV_CACHE_DIR=str(models_root / "other" / f"uv-{suffix}"),
        UV_PYTHON_INSTALL_DIR=str(models_root / "other" / f"uv-python-{suffix}"),
        UV_PROJECT_ENVIRONMENT=str(root / ".venv"),
        UV_PYTHON_PREFERENCE="only-managed",
        UV_PYTHON_DOWNLOADS="never",
        UV_LINK_MODE="copy",
        PULID_PROJECT_ROOT=str(root),
    )
    return environment


def patch_windows_cpu_backend(venv: Path) -> None:
    """Extract one hashed DLL; never install a second, competing distribution."""
    destination = venv / "Lib" / "site-packages" / "llama_cpp" / "lib"
    if not (destination / "ggml-cuda.dll").is_file():
        raise RuntimeError(f"Wheel llama-cpp CUDA incomplète : {destination}")
    with tempfile.TemporaryDirectory(dir=venv, prefix="pulid-dll-") as temporary:
        wheel = Path(temporary) / "cpu.whl"
        with urllib.request.urlopen(CPU_WHEEL_URL, timeout=120) as response, wheel.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != CPU_WHEEL_SHA256:
            raise RuntimeError(f"Empreinte wheel CPU incorrecte : {CPU_WHEEL_URL}")
        with zipfile.ZipFile(wheel) as archive:
            content = archive.read("llama_cpp/lib/ggml-cpu.dll")
        if hashlib.sha256(content).hexdigest() != CPU_DLL_SHA256:
            raise RuntimeError("Empreinte ggml-cpu.dll incorrecte dans la wheel CPU.")
        (destination / "ggml-cpu.dll").write_bytes(content)


def verify_lock(root: Path) -> None:
    manifest = json.loads((root / "runtime" / "lock-manifest.json").read_text())
    for name, expected_hash in manifest.items():
        path = root / name
        if hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest() != expected_hash:
            raise RuntimeError(f"Verrou incohérent : {path}. Récupérez une version complète des sources PuLID.")


def install(root: Path, models_root: Path, uv: Path, profile: str) -> None:
    verify_lock(root)
    python = verify_bootstrap(root, models_root)
    environment = clean_environment(root, models_root)
    expected_uv = (root / ".uv-version").read_text().strip()
    actual_uv = subprocess.check_output([str(uv), "--version"], text=True, env=environment)
    if actual_uv.split()[1] != expected_uv:
        raise RuntimeError(f"uv {expected_uv} requis : {uv}; reçu : {actual_uv.strip()}")
    if sys.platform == "darwin":
        manifest_path = root / "runtime" / "wheels" / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        wheel = manifest_path.parent / manifest["filename"]
        if not wheel.is_file():
            raise RuntimeError(f"Wheel Metal précompilée absente : {wheel}. Utilisez une archive PuLID complète.")
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError(f"Empreinte de la wheel Metal incorrecte : {wheel}.")
    venv = root / ".venv"
    # Do not let --clear traverse a user-provided junction to an unrelated folder.
    try:
        venv_metadata = venv.lstat()
    except FileNotFoundError:
        venv_metadata = None
    if venv.is_symlink() or (
        venv_metadata is not None
        and getattr(venv_metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise RuntimeError(f".venv est un lien/jonction : {venv}. Déplacez ce lien puis relancez l'installation.")

    def run(*arguments: str) -> None:
        subprocess.run([str(uv), *arguments], cwd=root, env=environment, check=True)

    run("venv", "--clear", "--python", str(python), str(venv))
    common = ("sync", "--frozen", "--python", str(python), "--no-default-groups")
    run(*common, "--only-group", "build", "--no-install-project")
    extras: list[str] = []
    for name in ("inference", "pulid", "server", "embeddings"):
        extras.extend(("--extra", name))
    if profile == "development":
        extras.extend(("--extra", "dev"))
    else:
        extras.append("--no-editable")
    run(*common, "--group", "build", "--no-build-isolation", *extras)
    venv_python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    run("pip", "check", "--python", str(venv_python))
    if sys.platform == "win32":
        patch_windows_cpu_backend(venv)
    state = {
        "project_root": str(root), "managed_python": str(python),
        "python": platform.python_version(), "uv": expected_uv, "profile": profile,
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
    }
    (venv / "pulid-runtime.json").write_text(json.dumps(state, indent=2) + "\n")
    print(f"Environnement vérifié : {venv_python}\nPython géré : {python}\nuv : {expected_uv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--profile", choices=("production", "development"), required=True)
    args = parser.parse_args()
    try:
        install(Path(__file__).resolve().parents[1], args.models_root.resolve(), args.uv.resolve(), args.profile)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERREUR] {exc}\nRelancez l'installateur de votre plateforme.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
