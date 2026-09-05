#!/usr/bin/env python3
"""Offline path repair and validation of the installed, managed PuLID runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import stat
import tempfile


def runtime_python(root: Path, state: dict[str, str]) -> Path:
    """Rebase only Python stored inside the project; external roots stay fixed."""
    if "managed_python_relative" in state:
        return root / state["managed_python_relative"]
    python = Path(state["managed_python"])
    try:
        return root / python.relative_to(state["project_root"])
    except ValueError:
        return python


def write_if_changed(path: Path, content: str | bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"Fichier runtime lié : {path}.")
    data = content.encode("utf-8") if isinstance(content, str) else content
    if path.exists() and path.read_bytes() == data:
        return
    # Concurrent frontend/backend starts must never read a half-written file.
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_state(root: Path, state: dict[str, str]) -> None:
    if state["python"] != (root / ".python-version").read_text().strip():
        raise ValueError("La version Python de l'installation est périmée.")
    if state["uv"] != (root / ".uv-version").read_text().strip():
        raise ValueError("La version uv de l'installation est périmée.")
    if state["lock_sha256"] != hashlib.sha256((root / "uv.lock").read_text(encoding="utf-8").encode("utf-8")).hexdigest():
        raise ValueError("Les dépendances verrouillées ont changé.")


def windows_venv_launchers(venv: Path, python: Path) -> dict[Path, bytes]:
    """Use CPython's bundled redirectors, which read home from pyvenv.cfg.

    uv 0.12.10 can embed an absolute minor-junction target in its trampoline.
    --relocatable and editing pyvenv.cfg do not repair that executable resource.
    Read both replacements before writing anything; never copy the base Python
    executable (it needs a different DLL layout) or use a system launcher.
    """
    bundled = python.parent / "Lib/venv/scripts/nt"
    replacements = {}
    for name, aliases in (
        ("python.exe", ("python3.exe", "python3.11.exe")),
        ("pythonw.exe", ()),
    ):
        source = bundled / name
        if not source.is_file():
            raise ValueError(f"Lanceur CPython géré absent : {source}. Réinstallez le Python géré PuLID.")
        content = source.read_bytes()
        replacements[venv / "Scripts" / name] = content
        for alias in aliases:
            target = venv / "Scripts" / alias
            if target.exists():
                replacements[target] = content
    return replacements


def prepare_environment(root: Path) -> None:
    """Repair local path metadata, never resolve or install dependencies."""
    venv = root / ".venv"
    metadata = venv.lstat()
    if venv.is_symlink() or getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f".venv est un lien/jonction : {venv}.")
    state = json.loads((venv / "pulid-runtime.json").read_text(encoding="utf-8"))
    validate_state(root, state)
    python = runtime_python(root, state)
    # This helper runs with the managed interpreter even if .venv is broken.
    # The venv interpreter is also accepted to migrate an installation in place.
    if (not python.is_file() or platform.python_version() != state["python"]
            or Path(sys._base_executable).resolve() != python.resolve()):
        raise ValueError(f"Python géré {state['python']} requis : {python}.")
    launchers = windows_venv_launchers(venv, python) if sys.platform == "win32" else {}
    config_path = venv / "pyvenv.cfg"
    config = dict(line.split(" = ", 1) for line in config_path.read_text(encoding="utf-8").splitlines() if " = " in line)
    config["home"] = str(python.parent)
    config["include-system-site-packages"] = "false"
    # uv normally writes only home; handle these optional CPython/venv keys too.
    if "executable" in config:
        config["executable"] = str(python)
    for key in ("base-prefix", "base-exec-prefix"):
        if key in config:
            config[key] = str(python.parent if sys.platform == "win32" else python.parent.parent)
    if "base-executable" in config:
        config["base-executable"] = str(python)
    write_if_changed(config_path, "".join(f"{key} = {value}\n" for key, value in config.items()))
    for executable, content in launchers.items():
        write_if_changed(executable, content)
    if sys.platform != "win32":
        executable = venv / "bin/python"
        target = os.path.relpath(python, executable.parent)
        if not executable.is_symlink() or os.readlink(executable) != target:
            # uv's remaining python3/python3.11 links already point at python.
            with tempfile.TemporaryDirectory(dir=executable.parent) as directory:
                link = Path(directory) / "python"
                link.symlink_to(target)
                os.replace(link, executable)
    if state["profile"] == "development":
        packages = venv / ("Lib/site-packages" if sys.platform == "win32" else "lib/python3.11/site-packages")
        # Hatchling's sole editable path is relative to site-packages, not CWD.
        write_if_changed(packages / "_pulid_app.pth", os.path.relpath(root / "src", packages) + "\n")
    try:
        state["managed_python_relative"] = str(python.relative_to(root))
    except ValueError:
        state.pop("managed_python_relative", None)
    state["project_root"] = str(root)
    state["managed_python"] = str(python)
    write_if_changed(venv / "pulid-python-path", state.get("managed_python_relative", str(python)) + "\n")
    write_if_changed(venv / "pulid-runtime.json", json.dumps(state, indent=2) + "\n")


def check_environment(root: Path) -> dict[str, str]:
    venv = root / ".venv"
    state_path = venv / "pulid-runtime.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        version = (root / ".python-version").read_text().strip()
        if Path(state["project_root"]) != root:
            raise ValueError("Le dossier PuLID a été déplacé.")
        if Path(sys.prefix).resolve() != venv.resolve() or platform.python_version() != version:
            raise ValueError(f"Interpréteur inattendu : {sys.executable} ({platform.python_version()}).")
        managed = Path(state["managed_python"])
        if not managed.is_file() or Path(sys._base_executable).resolve() != managed.resolve():
            raise ValueError(f"Python géré absent ou déplacé : {managed}.")
        validate_state(root, state)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Environnement PuLID inutilisable : {venv}. {exc} "
            "Après déplacement, démarrez avec start_windows.bat ou ./start_pulid_server.sh. "
            "Relancez install_windows.bat ou ./install_macos.sh depuis ce dossier."
        ) from exc
    return state


def main() -> int:
    try:
        root = Path(__file__).resolve().parents[1]
        if sys.argv[1:] == ["--prepare"]:
            prepare_environment(root)
            return 0
        state = check_environment(root)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[ERREUR] {exc} Vérifiez que le dossier complet a été déplacé ; sinon relancez l’installateur.", file=sys.stderr)
        return 1
    print(f"Python {state['python']} géré : {state['managed_python']}; uv {state['uv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
