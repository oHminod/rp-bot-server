"""Installation idempotente des modèles et configurations de PuLID."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import BinaryIO
from urllib.request import Request, urlopen
import zipfile

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)

from pulid_app.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PULID_REVISION,
    LOCAL_CONFIG_PATH,
    PROJECT_ROOT,
)
from pulid_app.models.pulid_assets import ensure_official_source
from pulid_app.paths import (
    ANTELOPEV2_REQUIRED_FILES,
    configure_external_model_caches,
    ensure_writable_directory,
)


MODELS_DIRECTORY_NAME = "PuLID_models"
INSIGHTFACE_LICENSE_URL = (
    "https://github.com/deepinsight/insightface/blob/master/server/LICENSING.md"
)
SDXL_CONFIG_REPOSITORY = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_CONFIG_DIRECTORY = "sdxl/stable-diffusion-xl-base-1.0-config"
SDXL_CONFIG_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "**/*.json",
    "**/*.txt",
    "**/*.model",
)
SDXL_FORBIDDEN_WEIGHT_SUFFIXES = frozenset(
    {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
)
SDXL_REQUIRED_CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "unet/config.json",
    "vae/config.json",
)


class InstallerError(RuntimeError):
    """L'installation ne peut pas produire un environnement fonctionnel."""


@dataclass(frozen=True)
class HTTPAsset:
    name: str
    relative_path: str
    url: str
    sha256: str


@dataclass(frozen=True)
class HuggingFaceAsset:
    name: str
    relative_path: str
    repository: str
    filename: str
    sha256: str
    revision: str = "main"


PULID_CHECKPOINT = HuggingFaceAsset(
    name="PuLID v1.1",
    relative_path="pulid_v1.1.safetensors",
    repository="guozinan/PuLID",
    filename="pulid_v1.1.safetensors",
    sha256="4cb8ceec1078e0165399b88332ab3c5971619111b8e1730e6bae64144aabae41",
)
BGE_CHECKPOINT = HuggingFaceAsset(
    name="BGE-M3 Q8_0",
    relative_path="text_embedding/bge-m3-Q8_0.gguf",
    repository="KimChen/bge-m3-GGUF",
    filename="bge-m3-q8_0.gguf",
    sha256="950f4a8e5e19477a6d3c26d2f162233c20002c601f75e4b002e3239997821167",
)
EVA_CLIP_REPOSITORY = "QuanSun/EVA-CLIP"
EVA_CLIP_FILENAME = "EVA02_CLIP_L_336_psz14_s6B.pt"
EVA_CLIP_REVISION = "11afd202f2ae80869d6cef18b1ec775e79bd8d12"
EVA_CLIP_SHA256 = "84c3a17a228c567a155259b2245b0b59072bf7da510260a0a02ec54de6d50b05"

FACEXLIB_ASSETS = (
    HTTPAsset(
        name="FaceXLib RetinaFace",
        relative_path="facexlib/weights/detection_Resnet50_Final.pth",
        url=(
            "https://github.com/xinntao/facexlib/releases/download/v0.1.0/"
            "detection_Resnet50_Final.pth"
        ),
        sha256="6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
    ),
    HTTPAsset(
        name="FaceXLib BiSeNet",
        relative_path="facexlib/weights/parsing_bisenet.pth",
        url=(
            "https://github.com/xinntao/facexlib/releases/download/v0.2.0/"
            "parsing_bisenet.pth"
        ),
        sha256="468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567",
    ),
    HTTPAsset(
        name="FaceXLib ParseNet",
        relative_path="facexlib/weights/parsing_parsenet.pth",
        url=(
            "https://github.com/xinntao/facexlib/releases/download/v0.2.2/"
            "parsing_parsenet.pth"
        ),
        sha256="3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2",
    ),
)

ANTELOPE_ARCHIVE = HTTPAsset(
    name="InsightFace AntelopeV2",
    relative_path="other/downloads/antelopev2.zip",
    url="https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
    sha256="8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5",
)
ANTELOPE_FILE_SHA256 = {
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "glintr100.onnx": "4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
    "scrfd_10g_bnkps.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
}

BASE_SDXL = HuggingFaceAsset(
    name="Stable Diffusion XL Base 1.0",
    relative_path="checkpoints/sd_xl_base_1.0.safetensors",
    repository=SDXL_CONFIG_REPOSITORY,
    filename="sd_xl_base_1.0.safetensors",
    sha256="31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
)


def resolve_models_root(
    selected: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Résout un dossier parent ou un dossier ``PuLID_models`` déjà nommé."""

    candidate = Path(selected).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve(strict=False)
    if candidate.name.casefold() == MODELS_DIRECTORY_NAME.casefold():
        return candidate
    return candidate / MODELS_DIRECTORY_NAME


def ask_yes_no(
    question: str,
    *,
    default: bool,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Pose une question oui/non et refuse les réponses ambiguës."""

    suffix = " [O/n] " if default else " [o/N] "
    while True:
        answer = input_fn(question + suffix).strip().casefold()
        if not answer:
            return default
        if answer in {"o", "oui", "y", "yes"}:
            return True
        if answer in {"n", "non", "no"}:
            return False


def prompt_models_root(
    *,
    project_root: Path = PROJECT_ROOT,
    input_fn: Callable[[str], str] = input,
) -> Path:
    """Demande l'emplacement, avec la racine du projet comme parent par défaut."""

    default = resolve_models_root(project_root, project_root=project_root)
    if ask_yes_no(
        f"Utiliser l'emplacement par défaut {default} ?",
        default=True,
        input_fn=input_fn,
    ):
        return default
    while True:
        raw = input_fn(
            "Chemin du dossier parent (ou d'un dossier déjà nommé PuLID_models) : "
        ).strip()
        if raw:
            return resolve_models_root(raw, project_root=project_root)


def read_local_installation(
    *,
    config_path: Path = LOCAL_CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path | None, Path | None]:
    """Lit les chemins persistés par une précédente installation."""

    if not config_path.is_file():
        return None, None
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None, None
    if not isinstance(raw, dict):
        return None, None

    raw_models_root = raw.get("models_root")
    if not isinstance(raw_models_root, str) or not raw_models_root.strip():
        return None, None
    models_root = Path(raw_models_root.strip()).expanduser()
    if not models_root.is_absolute():
        models_root = project_root / models_root
    models_root = models_root.resolve(strict=False)

    raw_sdxl = raw.get("sdxl")
    raw_checkpoint = (
        raw_sdxl.get("checkpoint") if isinstance(raw_sdxl, dict) else None
    )
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint.strip():
        return models_root, None
    checkpoint = Path(raw_checkpoint.strip()).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = models_root / checkpoint
    return models_root, checkpoint.resolve(strict=False)


def find_existing_models_root(
    *,
    project_root: Path = PROJECT_ROOT,
    config_path: Path = LOCAL_CONFIG_PATH,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Retrouve une racine existante sans poser de question interactive."""

    environment = os.environ if environ is None else environ
    candidates: list[Path] = []
    environment_root = environment.get("PULID_MODELS_ROOT", "").strip()
    if environment_root:
        candidates.append(
            resolve_models_root(environment_root, project_root=project_root)
        )
    configured_root, _checkpoint = read_local_installation(
        config_path=config_path,
        project_root=project_root,
    )
    if configured_root is not None:
        candidates.append(configured_root)
    candidates.append(resolve_models_root(project_root, project_root=project_root))

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, expected_sha256: str) -> bool:
    return path.is_file() and _sha256(path) == expected_sha256


def _replace_download(downloaded: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve(strict=False) == target.resolve(strict=False):
        return
    os.replace(downloaded, target)


def ensure_huggingface_asset(
    models_root: Path,
    asset: HuggingFaceAsset,
    console: Console,
    *,
    downloader: Callable[..., str] | None = None,
) -> Path:
    """Valide ou télécharge un fichier Hugging Face dans sa destination finale."""

    target = models_root / asset.relative_path
    if _matches(target, asset.sha256):
        console.print(f"[green]✓[/] {asset.name} déjà présent")
        return target
    if target.exists():
        console.print(f"[yellow]↻[/] {asset.name} est incomplet ou corrompu, réparation")
    else:
        console.print(f"[cyan]↓[/] Téléchargement de {asset.name}")
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(
            downloader(
                repo_id=asset.repository,
                filename=asset.filename,
                revision=asset.revision,
                local_dir=target.parent,
                cache_dir=models_root / "huggingface" / "hub",
                force_download=target.exists(),
            )
        )
        _replace_download(downloaded, target)
    except Exception as exc:
        raise InstallerError(
            f"Téléchargement de {asset.name} impossible vers {target} : {exc}"
        ) from exc
    if not _matches(target, asset.sha256):
        raise InstallerError(
            f"Empreinte SHA-256 invalide pour {asset.name} : {target}. "
            "Relancez l'installation pour retélécharger le fichier."
        )
    return target


def _open_url(url: str) -> BinaryIO:
    request = Request(url, headers={"User-Agent": "PuLID-installer/0.1"})
    return urlopen(request, timeout=60)


def download_http(
    url: str,
    destination: Path,
    console: Console,
    *,
    opener: Callable[[str], BinaryIO] = _open_url,
) -> None:
    """Télécharge un fichier vers un ``.part`` puis le publie atomiquement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        with opener(url) as source, temporary.open("wb") as output:
            raw_total = getattr(source, "headers", {}).get("Content-Length")
            total = int(raw_total) if raw_total and raw_total.isdigit() else None
            with Progress(
                TextColumn("  {task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(destination.name, total=total)
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
                    progress.update(task, advance=len(chunk))
        os.replace(temporary, destination)
    except Exception as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise InstallerError(f"Téléchargement impossible depuis {url} : {exc}") from exc


def ensure_http_asset(
    models_root: Path,
    asset: HTTPAsset,
    console: Console,
    *,
    downloader: Callable[[str, Path, Console], None] = download_http,
) -> Path:
    target = models_root / asset.relative_path
    if _matches(target, asset.sha256):
        console.print(f"[green]✓[/] {asset.name} déjà présent")
        return target
    action = "réparation" if target.exists() else "téléchargement"
    console.print(f"[cyan]↓[/] {asset.name} — {action}")
    downloader(asset.url, target, console)
    if not _matches(target, asset.sha256):
        raise InstallerError(f"Empreinte SHA-256 invalide pour {asset.name} : {target}")
    return target


def _antelope_is_ready(models_root: Path) -> bool:
    directory = models_root / "antelopev2"
    return all(
        _matches(directory / name, ANTELOPE_FILE_SHA256[name])
        for name in ANTELOPEV2_REQUIRED_FILES
    )


def confirm_antelope_license(
    models_root: Path,
    console: Console,
    *,
    accepted: bool = False,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Exige l'acceptation de la licence avant le téléchargement d'AntelopeV2."""

    if _antelope_is_ready(models_root):
        return True
    console.print()
    console.print("[bold yellow]Licence des modèles InsightFace AntelopeV2[/]")
    console.print(
        "Les poids AntelopeV2 sont fournis uniquement pour la recherche "
        "non commerciale. Leur usage commercial exige une licence distincte."
    )
    console.print(f"Conditions officielles : {INSIGHTFACE_LICENSE_URL}")
    if accepted:
        return False
    if not ask_yes_no(
        "Acceptez-vous ces conditions pour télécharger AntelopeV2 ?",
        default=False,
        input_fn=input_fn,
    ):
        raise InstallerError(
            "La licence InsightFace n'a pas été acceptée ; "
            "AntelopeV2 ne sera pas téléchargé."
        )
    return False


def install_antelope_archive(models_root: Path, archive_path: Path) -> None:
    """Extrait uniquement les cinq ONNX attendus et vérifie chaque empreinte."""

    destination = models_root / "antelopev2"
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for name in sorted(ANTELOPEV2_REQUIRED_FILES):
                member = f"antelopev2/{name}"
                target = destination / name
                temporary = target.with_name(target.name + ".part")
                with archive.open(member) as source, temporary.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                if not _matches(temporary, ANTELOPE_FILE_SHA256[name]):
                    temporary.unlink(missing_ok=True)
                    raise InstallerError(
                        f"Empreinte invalide pour {member} dans {archive_path}."
                    )
                os.replace(temporary, target)
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise InstallerError(f"Archive AntelopeV2 invalide : {archive_path} ({exc})") from exc


def ensure_antelope(
    models_root: Path,
    console: Console,
    *,
    already_ready: bool | None = None,
) -> Path:
    ready = (
        already_ready
        if already_ready is not None
        else _antelope_is_ready(models_root)
    )
    if ready:
        console.print("[green]✓[/] InsightFace AntelopeV2 déjà présent")
        return models_root / "antelopev2"
    archive = ensure_http_asset(models_root, ANTELOPE_ARCHIVE, console)
    console.print("[cyan]↻[/] Installation des modèles ONNX AntelopeV2")
    install_antelope_archive(models_root, archive)
    if not _antelope_is_ready(models_root):
        raise InstallerError("AntelopeV2 reste incomplet après extraction.")
    return models_root / "antelopev2"


def ensure_eva_clip(models_root: Path, console: Console) -> Path:
    snapshots = (
        models_root
        / "huggingface"
        / "hub"
        / "models--QuanSun--EVA-CLIP"
        / "snapshots"
    )
    for candidate in snapshots.glob(f"*/{EVA_CLIP_FILENAME}"):
        if _matches(candidate, EVA_CLIP_SHA256):
            console.print("[green]✓[/] EVA-CLIP déjà présent")
            return candidate
    console.print("[cyan]↓[/] Téléchargement d'EVA-CLIP")
    try:
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(
                repo_id=EVA_CLIP_REPOSITORY,
                filename=EVA_CLIP_FILENAME,
                revision=EVA_CLIP_REVISION,
                cache_dir=models_root / "huggingface" / "hub",
            )
        )
    except Exception as exc:
        raise InstallerError(f"Téléchargement EVA-CLIP impossible : {exc}") from exc
    if not _matches(downloaded, EVA_CLIP_SHA256):
        raise InstallerError(f"Empreinte EVA-CLIP invalide : {downloaded}")
    return downloaded


def validate_sdxl_config_tree(path: Path) -> tuple[int, int]:
    missing = [
        path / relative
        for relative in SDXL_REQUIRED_CONFIG_FILES
        if not (path / relative).is_file()
    ]
    if missing:
        raise InstallerError(
            "Configuration SDXL incomplète ; fichiers manquants : "
            + ", ".join(str(file) for file in missing)
        )
    files = [file for file in path.rglob("*") if file.is_file()]
    forbidden = [
        file
        for file in files
        if file.suffix.casefold() in SDXL_FORBIDDEN_WEIGHT_SUFFIXES
    ]
    if forbidden:
        raise InstallerError(
            "Des poids interdits ont été trouvés dans le dossier réservé aux "
            "configurations SDXL : "
            + ", ".join(str(file) for file in forbidden)
        )
    config_files = [
        file for file in files if ".cache" not in file.relative_to(path).parts
    ]
    return len(config_files), sum(file.stat().st_size for file in config_files)


def ensure_sdxl_configuration(
    models_root: Path,
    console: Console,
    *,
    force: bool = False,
) -> Path:
    destination = models_root / SDXL_CONFIG_DIRECTORY
    console.print("[cyan]↻[/] Vérification des configurations et tokenizers SDXL")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=SDXL_CONFIG_REPOSITORY,
            cache_dir=models_root / "huggingface" / "hub",
            local_dir=destination,
            allow_patterns=list(SDXL_CONFIG_PATTERNS),
            force_download=force,
        )
        validate_sdxl_config_tree(destination)
    except InstallerError:
        raise
    except Exception as exc:
        if not force:
            try:
                validate_sdxl_config_tree(destination)
            except InstallerError:
                pass
            else:
                console.print(
                    "[yellow]⚠[/] Mise à jour SDXL indisponible ; "
                    "la configuration locale complète est conservée."
                )
                return destination
        raise InstallerError(
            f"Préparation de la configuration SDXL impossible : {exc}"
        ) from exc
    return destination


def discover_sdxl_checkpoints(models_root: Path) -> tuple[Path, ...]:
    checkpoints = models_root / "checkpoints"
    if not checkpoints.is_dir():
        return ()
    return tuple(
        sorted(
            (path for path in checkpoints.glob("*.safetensors") if path.is_file()),
            key=lambda path: path.name.casefold(),
        )
    )


def choose_checkpoint(
    candidates: Sequence[Path],
    *,
    input_fn: Callable[[str], str] = input,
) -> Path:
    if not candidates:
        raise InstallerError("Aucun checkpoint SDXL .safetensors n'a été détecté.")
    if len(candidates) == 1:
        return candidates[0]
    print("Checkpoints SDXL détectés :")
    for index, candidate in enumerate(candidates, start=1):
        print(f"  {index}. {candidate.name}")
    while True:
        raw = input_fn("Numéro du checkpoint à utiliser [1] : ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]


def select_sdxl_checkpoint(
    models_root: Path,
    console: Console,
    *,
    mode: str = "ask",
    preferred_checkpoint: Path | None = None,
    input_fn: Callable[[str], str] = input,
) -> Path | None:
    checkpoints_dir = models_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    candidates = discover_sdxl_checkpoints(models_root)

    if mode == "skip":
        return None

    if mode == "ask" and candidates:
        resolved_preference = (
            preferred_checkpoint.resolve(strict=False)
            if preferred_checkpoint is not None
            else None
        )
        checkpoint = next(
            (
                candidate
                for candidate in candidates
                if candidate.resolve(strict=False) == resolved_preference
            ),
            candidates[0],
        )
        console.print(
            f"[green]✓[/] Checkpoint SDXL déjà présent : [cyan]{checkpoint}[/]"
        )
        return checkpoint

    if mode == "ask" and not ask_yes_no(
        "Souhaitez-vous installer un modèle SDXL maintenant ? "
        "Vous pourrez le faire ultérieurement en relançant l'installateur.",
        default=True,
        input_fn=input_fn,
    ):
        console.print(
            "[yellow]Installation du checkpoint SDXL différée.[/] "
            "Les autres composants vont être installés normalement."
        )
        return None

    has_model = mode == "existing"
    if mode == "ask":
        has_model = ask_yes_no(
            "Avez-vous déjà un modèle SDXL au format .safetensors ?",
            default=bool(candidates),
            input_fn=input_fn,
        )
    if has_model:
        console.print(
            "Déposez votre checkpoint SDXL dans :\n"
            f"  [cyan]{checkpoints_dir}[/]"
        )
        if not candidates and mode == "ask":
            input_fn("Appuyez sur Entrée après avoir copié le fichier : ")
            candidates = discover_sdxl_checkpoints(models_root)
            if not candidates:
                console.print(
                    "[yellow]Aucun checkpoint détecté ; installation SDXL "
                    "différée.[/]"
                )
                return None
        return choose_checkpoint(candidates, input_fn=input_fn)

    if mode == "ask" and not ask_yes_no(
        "Télécharger le modèle officiel Stable Diffusion XL Base 1.0 (~6,9 Go) ?",
        default=True,
        input_fn=input_fn,
    ):
        console.print(
            "[yellow]Téléchargement du checkpoint SDXL différé.[/] "
            "Relancez l'installateur lorsque vous souhaiterez l'ajouter."
        )
        return None
    return ensure_huggingface_asset(models_root, BASE_SDXL, console)


def _platform_device() -> str:
    if sys.platform == "darwin":
        return "mps"
    if sys.platform == "win32":
        return "cuda"
    return "cpu"


def write_local_config(
    models_root: Path,
    sdxl_checkpoint: Path | None,
    *,
    default_config: Path = DEFAULT_CONFIG_PATH,
    destination: Path = LOCAL_CONFIG_PATH,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Régénère la configuration locale depuis les valeurs à jour du dépôt."""

    try:
        raw = yaml.safe_load(default_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise InstallerError(
            f"Configuration par défaut illisible : {default_config} ({exc})"
        ) from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("sdxl"), dict):
        raise InstallerError(f"Configuration par défaut invalide : {default_config}")
    try:
        raw["models_root"] = models_root.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        raw["models_root"] = models_root.as_posix()
    if sdxl_checkpoint is not None:
        try:
            relative_checkpoint = sdxl_checkpoint.resolve(strict=False).relative_to(
                models_root.resolve(strict=False)
            )
        except ValueError as exc:
            raise InstallerError(
                f"Le checkpoint SDXL doit rester sous {models_root} : "
                f"{sdxl_checkpoint}"
            ) from exc
        raw["sdxl"]["checkpoint"] = relative_checkpoint.as_posix()
    device = raw.setdefault("device", {})
    if isinstance(device, dict):
        device["preferred"] = _platform_device()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    text = (
        "# Généré par pulid-install. Relancez l'installateur pour le mettre à jour.\n"
        + yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as exc:
        raise InstallerError(f"Impossible d'écrire {destination} : {exc}") from exc
    return destination


def prepare_required_assets(
    models_root: Path,
    console: Console,
    *,
    force_configs: bool = False,
    antelope_ready: bool | None = None,
) -> None:
    ensure_huggingface_asset(models_root, PULID_CHECKPOINT, console)
    ensure_antelope(models_root, console, already_ready=antelope_ready)
    for asset in FACEXLIB_ASSETS:
        ensure_http_asset(models_root, asset, console)
    ensure_huggingface_asset(models_root, BGE_CHECKPOINT, console)
    ensure_eva_clip(models_root, console)
    source = models_root / "sources" / "PuLID"
    try:
        result = ensure_official_source(
            source,
            DEFAULT_PULID_REVISION,
            repair_existing=True,
        )
    except Exception as exc:
        raise InstallerError(
            f"Préparation du code officiel PuLID impossible : {exc}"
        ) from exc
    action = "téléchargé" if result.downloaded else "déjà présent"
    console.print(f"[green]✓[/] Code officiel PuLID {action}")
    ensure_sdxl_configuration(models_root, console, force=force_configs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Installe ou répare les modèles et configurations nécessaires à PuLID."
        )
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        help=(
            "Dossier parent ou dossier déjà nommé PuLID_models. Sans cette option, "
            "le CLI réutilise l'installation détectée ou pose la question."
        ),
    )
    parser.add_argument(
        "--sdxl",
        choices=("ask", "existing", "download", "skip"),
        default="ask",
        help=(
            "Sélection interactive, modèle local existant, SDXL Base officiel "
            "ou installation différée."
        ),
    )
    parser.add_argument(
        "--force-configs",
        action="store_true",
        help="Retélécharge les petits fichiers de configuration SDXL.",
    )
    parser.add_argument(
        "--accept-insightface-license",
        action="store_true",
        help=(
            "Accepte sans question la licence non commerciale des modèles "
            "InsightFace AntelopeV2."
        ),
    )
    return parser


def run_installation(args: argparse.Namespace, console: Console) -> int:
    configured_root, configured_checkpoint = read_local_installation()
    if args.models_root is not None:
        models_root = resolve_models_root(args.models_root)
    else:
        models_root = find_existing_models_root() or prompt_models_root()
    ensure_writable_directory(models_root)
    configure_external_model_caches(models_root)
    for relative in (
        "checkpoints",
        "huggingface/hub",
        "huggingface/transformers",
        "torch",
        "other",
        "text_embedding",
    ):
        (models_root / relative).mkdir(parents=True, exist_ok=True)

    console.print(f"Racine des modèles : [bold cyan]{models_root}[/]")
    antelope_ready = confirm_antelope_license(
        models_root,
        console,
        accepted=args.accept_insightface_license,
    )
    preferred_checkpoint = (
        configured_checkpoint
        if configured_root == models_root and configured_checkpoint is not None
        else None
    )
    checkpoint = select_sdxl_checkpoint(
        models_root,
        console,
        mode=args.sdxl,
        preferred_checkpoint=preferred_checkpoint,
    )
    prepare_required_assets(
        models_root,
        console,
        force_configs=args.force_configs,
        antelope_ready=antelope_ready,
    )
    local_config = write_local_config(models_root, checkpoint)
    console.print()
    console.print("[bold green]Installation des modèles terminée.[/]")
    if checkpoint is None:
        console.print(
            "[yellow]Checkpoint SDXL : installation différée.[/] "
            f"Vous pourrez déposer un fichier .safetensors dans "
            f"[cyan]{models_root / 'checkpoints'}[/] puis relancer l'installateur."
        )
    else:
        console.print(f"Checkpoint SDXL : [cyan]{checkpoint}[/]")
    console.print(f"Configuration locale : [cyan]{local_config}[/]")
    console.print(
        "AntelopeV2 est distribué pour la recherche non commerciale ; "
        f"conditions : {INSIGHTFACE_LICENSE_URL}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        return run_installation(args, console)
    except (InstallerError, OSError, PermissionError) as exc:
        console.print(f"[bold red]Installation impossible :[/] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Installation annulée.[/]")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
