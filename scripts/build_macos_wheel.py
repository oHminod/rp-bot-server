#!/usr/bin/env python3
"""Maintainer: build the Metal wheel bundled in PuLID release archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

SOURCE_URL = "https://files.pythonhosted.org/packages/1a/71/5ecdad726b87527b361d9d95313c6591f27bb3fad79c9ab524ad4250b4c9/llama_cpp_python-0.3.35.tar.gz"
SOURCE_SHA256 = "1139dbb54509074b70893fab8554e3b079aa9f4d312058ce4018ef0019e3de12"
WHEEL_NAME = "llama_cpp_python-0.3.35-py3-none-macosx_11_0_arm64.whl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("La construction Metal requiert un Mac Apple Silicon avec Xcode/Command Line Tools.")
    if platform.python_version() != (root / ".python-version").read_text().strip():
        parser.error("Exécutez ce script avec le Python exact de .python-version.")
    uv = args.uv.resolve()
    if subprocess.check_output([str(uv), "--version"], text=True).split()[1] != (root / ".uv-version").read_text().strip():
        parser.error("Utilisez la version uv définie dans .uv-version.")
    output = root / "runtime" / "wheels"
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pulid-metal-build-") as temporary:
        work = Path(temporary)
        environment = {k: v for k, v in os.environ.items() if not k.startswith(("UV_", "CMAKE_", "SKBUILD_"))}
        environment.update(UV_CACHE_DIR=str(work / "cache"), MACOSX_DEPLOYMENT_TARGET="11.0", SOURCE_DATE_EPOCH="1788566400")
        python = work / "venv" / "bin" / "python"
        subprocess.run([str(uv), "venv", "--python", sys.executable, str(work / "venv")], env=environment, check=True)
        subprocess.run([str(uv), "pip", "sync", "--python", str(python), "--require-hashes", str(root / "requirements/wheel-build.txt")], env=environment, check=True)
        environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
        archive = work / "source.tar.gz"
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError("Empreinte incorrecte pour la source llama-cpp-python.")
        with tarfile.open(archive) as source:
            source.extractall(work / "source", filter="data")
        source_root = work / "source" / "llama_cpp_python-0.3.35"
        subprocess.run([
            str(uv), "build", str(source_root), "--wheel", "--python", str(python),
            "--no-build-isolation", "--out-dir", str(work / "wheel-output"),
            "-Ccmake.define.CMAKE_OSX_ARCHITECTURES=arm64",
            "-Ccmake.define.CMAKE_OSX_DEPLOYMENT_TARGET=11.0",
            "-Ccmake.define.GGML_NATIVE=OFF", "-Ccmake.define.GGML_METAL=ON",
            "-Ccmake.define.GGML_METAL_EMBED_LIBRARY=ON",
            "-Ccmake.define.GGML_ACCELERATE=ON",
        ], env=environment, check=True)
        shutil.copy2(work / "wheel-output" / WHEEL_NAME, output / WHEEL_NAME)
    return record_wheel(output)


def record_wheel(output: Path) -> int:
    wheel = output / WHEEL_NAME
    with zipfile.ZipFile(wheel) as built:
        if built.testzip() is not None:
            raise RuntimeError(f"Wheel ZIP invalide : {wheel}")
    manifest = {
        "filename": WHEEL_NAME, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
        "python": platform.python_version(), "platform": platform.platform(),
        "compiler": subprocess.check_output(["/usr/bin/clang", "--version"], text=True).strip(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(wheel)
    print(manifest["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
