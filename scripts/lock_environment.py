#!/usr/bin/env python3
"""Maintainer: resolve uv.lock, then bind it to source metadata and runtime pins."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

LOCK_INPUTS = ("pyproject.toml", "uv.lock", ".python-version", ".uv-version", "runtime/wheels/manifest.json")


def write_lock_manifest(root: Path) -> None:
    manifest = {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in LOCK_INPUTS}
    (root / "runtime" / "lock-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    subprocess.run([str(args.uv.resolve()), "lock", "--managed-python", "--python", (root / ".python-version").read_text().strip()], cwd=root, check=True)
    write_lock_manifest(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
