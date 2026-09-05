#!/usr/bin/env python3
"""Construit l'archive source PuLID déterministe et son empreinte SHA-256."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import tomllib


ARCHIVE_FORMAT_VERSION = 1
MODEL_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors"}
)
EXCLUDED_PATH_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "htmlcov",
        "node_modules",
        "tests",
        "venv",
    }
)
ROOT_FILES = (
    ".env.example",
    ".python-version",
    ".uv-version",
    "uv.lock",
    "WINDOWS_INSTALL_TEST.md",
    "API_FRONTEND_INTEGRATION.md",
    "LICENSE",
    "README.md",
    "RELEASE.md",
    "RP_BOT_TEXT_EMBEDDING_INTEGRATION.md",
    "install_macos.sh",
    "install_production_macos.sh",
    "install_production_windows.bat",
    "install_windows.bat",
    "pyproject.toml",
    "start_frontend_macos.sh",
    "start_frontend_windows.bat",
    "start_pulid_server.sh",
    "start_windows.bat",
)
STATIC_FILES = (
    "cache/identity/.gitkeep",
    "config/default.yaml",
    "inputs/.gitkeep",
    "outputs/.gitkeep",
)
TREE_DIRECTORIES = ("frontend", "src", "runtime", "requirements")
SCRIPT_NAMES = (
    "build_release.py",
    "bootstrap_windows.ps1",
    "install_environment.py",
    "check_environment.py",
    "build_macos_wheel.py",
    "lock_environment.py",
    "cache_identity.py",
    "inspect_models.py",
    "prepare_pulid.py",
    "prepare_sdxl_config.py",
    "verify_text_embedding.py",
)
REQUIRED_RELEASE_FILES = frozenset(
    {
        "README.md",
        "config/default.yaml",
        "install_macos.sh",
        "install_production_macos.sh",
        "install_production_windows.bat",
        "install_windows.bat",
        "pyproject.toml",
        "uv.lock",
        ".uv-version",
        "runtime/lock-manifest.json",
        "runtime/wheels/manifest.json",
        "scripts/bootstrap_windows.ps1",
        "scripts/install_environment.py",
        "scripts/check_environment.py",
        "src/pulid_app/__init__.py",
        "src/pulid_app/api_contract.py",
        "start_pulid_server.sh",
        "start_windows.bat",
    }
)


def read_project_version(project_root: Path) -> str:
    """Lit la version applicative depuis son unique source, ``pyproject.toml``."""

    pyproject = project_root / "pyproject.toml"
    try:
        parsed = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        version = parsed["project"]["version"]
    except (KeyError, OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(f"Version PuLID illisible dans {pyproject}: {exc}") from exc
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Version PuLID invalide dans {pyproject}: {version!r}")
    return version.strip()


def read_api_contract_version(project_root: Path) -> str:
    """Lit la constante du contrat sans importer le serveur ni ses dépendances."""

    module_path = project_root / "src" / "pulid_app" / "api_contract.py"
    try:
        module = ast.parse(module_path.read_text(encoding="utf-8"), module_path.name)
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"Contrat API illisible dans {module_path}: {exc}") from exc
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "API_CONTRACT_VERSION"
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RuntimeError(f"API_CONTRACT_VERSION absent ou invalide dans {module_path}")


def _is_release_file(relative: Path) -> bool:
    posix = relative.as_posix()
    if relative.suffix.casefold() in MODEL_SUFFIXES:
        return False
    if relative.name in {".DS_Store", "Thumbs.db"}:
        return False
    if relative.name == ".env" or relative.name.startswith("test_"):
        return False
    if relative.name == "local.yaml" or relative.name.endswith(".local.yaml"):
        return False
    if any(part in EXCLUDED_PATH_PARTS for part in relative.parts):
        return False
    if posix in ROOT_FILES or posix in STATIC_FILES:
        return True
    if relative.parts[0] in TREE_DIRECTORIES:
        return True
    return (
        len(relative.parts) == 2
        and relative.parts[0] == "scripts"
        and relative.name in SCRIPT_NAMES
    )


def collect_release_files(project_root: Path) -> tuple[Path, ...]:
    """Retourne la liste blanche triée des fichiers distribuables."""

    root = project_root.resolve()
    # A source checkout alone cannot produce an installable Mac archive.
    try:
        from scripts.install_environment import verify_lock
    except ModuleNotFoundError:
        from install_environment import verify_lock
    verify_lock(root)
    candidates = [root / name for name in (*ROOT_FILES, *STATIC_FILES)]
    candidates.extend(root / "scripts" / name for name in SCRIPT_NAMES)
    for name in TREE_DIRECTORIES:
        candidates.extend((root / name).rglob("*"))
    selected = tuple(
        sorted(
            (
                path
                for path in candidates
                if path.is_file() and _is_release_file(path.relative_to(root))
            ),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    wheel_manifest = root / "runtime/wheels/manifest.json"
    if wheel_manifest.is_file():
        manifest = json.loads(wheel_manifest.read_text())
        wheel = wheel_manifest.parent / manifest["filename"]
        if not wheel.is_file() or sha256_file(wheel) != manifest["sha256"]:
            raise RuntimeError(f"Wheel Metal absente ou corrompue : {wheel}. Exécutez scripts/build_macos_wheel.py puis scripts/lock_environment.py.")
    relative_files = {path.relative_to(root).as_posix() for path in selected}
    missing = sorted(REQUIRED_RELEASE_FILES - relative_files)
    if missing:
        raise RuntimeError(
            "Archive PuLID incomplète ; fichiers requis absents : " + ", ".join(missing)
        )
    return selected


def _tar_info(name: str, *, directory: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.mode = 0o755 if directory or name.endswith(".sh") else 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    return info


def _release_metadata(version: str, api_contract_version: str) -> bytes:
    payload = {
        "apiContractVersion": api_contract_version,
        "archiveFormatVersion": ARCHIVE_FORMAT_VERSION,
        "component": "pulid",
        "version": version,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_archive(
    project_root: Path,
    destination: Path,
    *,
    version: str,
    api_contract_version: str,
) -> None:
    prefix = PurePosixPath(f"pulid-{version}")
    files = collect_release_files(project_root)
    directories = {prefix}
    for path in files:
        relative = PurePosixPath(path.relative_to(project_root).as_posix())
        parent = relative.parent
        while parent != PurePosixPath("."):
            directories.add(prefix / parent)
            parent = parent.parent

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=temporary,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.USTAR_FORMAT,
                ) as archive:
                    for directory in sorted(directories, key=str):
                        archive.addfile(_tar_info(str(directory) + "/", directory=True))

                    metadata = _release_metadata(version, api_contract_version)
                    metadata_info = _tar_info(
                        str(prefix / "release-metadata.json"), directory=False
                    )
                    metadata_info.size = len(metadata)
                    archive.addfile(metadata_info, io.BytesIO(metadata))

                    for path in files:
                        relative = path.relative_to(project_root).as_posix()
                        content = path.read_bytes()
                        info = _tar_info(str(prefix / relative), directory=False)
                        info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))
            os.replace(temporary_path, destination)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_release(project_root: Path, output_dir: Path) -> tuple[Path, Path]:
    """Construit l'archive et un fichier de checksum référençant cet artefact."""

    root = project_root.resolve()
    version = read_project_version(root)
    api_contract_version = read_api_contract_version(root)
    destination = output_dir.resolve() / f"pulid-{version}.tar.gz"
    _write_archive(
        root,
        destination,
        version=version,
        api_contract_version=api_contract_version,
    )
    checksum_path = destination.parent / "checksums-sha256.txt"
    checksum_path.write_text(
        f"{sha256_file(destination)}  {destination.name}\n",
        encoding="ascii",
        newline="\n",
    )
    return destination, checksum_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construit l'archive PuLID déterministe et son SHA-256."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="Dossier de sortie (défaut : dist).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    archive_path, checksum_path = build_release(project_root, args.output_dir)
    print(f"Archive : {archive_path}")
    print(f"SHA-256 : {sha256_file(archive_path)}")
    print(f"Checksums : {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
