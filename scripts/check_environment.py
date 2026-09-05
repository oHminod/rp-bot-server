#!/usr/bin/env python3
"""Fast stdlib-only guard: detect moved, stale or foreign PuLID environments."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys


def check_environment(root: Path) -> dict[str, str]:
    venv = root / ".venv"
    state_path = venv / "pulid-runtime.json"
    try:
        state = json.loads(state_path.read_text())
        version = (root / ".python-version").read_text().strip()
        if Path(state["project_root"]) != root:
            raise ValueError("Le dossier PuLID a été déplacé.")
        if Path(sys.prefix).resolve() != venv.resolve() or platform.python_version() != version:
            raise ValueError(f"Interpréteur inattendu : {sys.executable} ({platform.python_version()}).")
        managed = Path(state["managed_python"])
        if not managed.is_file() or Path(sys._base_executable).resolve() != managed.resolve():
            raise ValueError(f"Python géré absent ou déplacé : {managed}.")
        if state["uv"] != (root / ".uv-version").read_text().strip():
            raise ValueError("La version uv de l'installation est périmée.")
        if state["lock_sha256"] != hashlib.sha256((root / "uv.lock").read_text(encoding="utf-8").encode("utf-8")).hexdigest():
            raise ValueError("Les dépendances verrouillées ont changé.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Environnement PuLID à recréer : {venv}. {exc} "
            "Relancez install_windows.bat ou ./install_macos.sh depuis ce dossier."
        ) from exc
    return state


def main() -> int:
    try:
        state = check_environment(Path(__file__).resolve().parents[1])
    except RuntimeError as exc:
        print(f"[ERREUR] {exc}", file=sys.stderr)
        return 1
    print(f"Python {state['python']} géré : {state['managed_python']}; uv {state['uv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
