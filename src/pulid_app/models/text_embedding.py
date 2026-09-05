"""Embeddings de texte GGUF locaux sur CPU, Metal ou CUDA."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any

from pulid_app.config import TextEmbeddingConfig
from pulid_app.exceptions import EmbeddingError, ModelLoadError, ModelNotFoundError


MAX_EMBEDDING_INPUTS = 64
MAX_EMBEDDING_INPUT_CHARS = 32_000
MAX_EMBEDDING_TOTAL_CHARS = 128_000
SUPPORTED_EMBEDDING_DEVICES = frozenset({"cpu", "cuda", "mps"})


def _embedding_device_type(device: str) -> str:
    normalized = device.strip().casefold().split(":", maxsplit=1)[0]
    if normalized not in SUPPORTED_EMBEDDING_DEVICES:
        supported = ", ".join(sorted(SUPPORTED_EMBEDDING_DEVICES))
        raise ValueError(
            f"Device d'embedding inconnu : {device!r}. Valeurs acceptées : {supported}."
        )
    return normalized


def _cuda_dll_candidates(
    *,
    prefix: Path,
    environ: Mapping[str, str],
    torch_package_dir: Path | None,
    windows_root: Path | None = None,
) -> tuple[Path, ...]:
    candidates = [prefix / "Lib" / "site-packages" / "torch" / "lib"]
    if torch_package_dir is not None:
        candidates.append(torch_package_dir / "lib")
    for name, value in environ.items():
        normalized_name = name.upper()
        if normalized_name == "CUDA_PATH" or normalized_name.startswith(
            "CUDA_PATH_V"
        ):
            if value.strip():
                candidates.append(Path(value) / "bin")
    if windows_root is not None:
        candidates.extend(_nvidia_driver_cuda_dll_directories(windows_root))

    selected: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        key = str(resolved).casefold()
        if key not in seen and resolved.is_dir():
            selected.append(resolved)
            seen.add(key)
    return tuple(selected)


def _nvidia_driver_cuda_dll_directories(windows_root: Path) -> tuple[Path, ...]:
    """Trouve le runtime CUDA 13 livré avec le pilote NVIDIA Windows."""

    repository = windows_root / "System32" / "DriverStore" / "FileRepository"
    if not repository.is_dir():
        return ()
    try:
        runtime_dlls = list(repository.glob("*/nvcudart_hybrid64.dll"))
    except OSError:
        return ()

    def modification_time(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    runtime_dlls.sort(key=modification_time, reverse=True)
    return (runtime_dlls[0].parent,) if runtime_dlls else ()


def _prepend_dll_directories_to_path(
    current_path: str,
    directories: Sequence[Path],
    *,
    separator: str,
) -> str:
    """Préfixe PATH sans dupliquer les dossiers déjà présents."""

    entries = [str(directory) for directory in directories]
    entries.extend(part for part in current_path.split(separator) if part)
    selected: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        key = entry.casefold()
        if key not in seen:
            selected.append(entry)
            seen.add(key)
    return separator.join(selected)


def _windows_cuda_dll_directories() -> tuple[Path, ...]:
    if os.name != "nt":
        return ()
    torch_spec = importlib.util.find_spec("torch")
    torch_package_dir = (
        Path(torch_spec.origin).parent
        if torch_spec is not None and torch_spec.origin is not None
        else None
    )
    return _cuda_dll_candidates(
        prefix=Path(sys.prefix),
        environ=os.environ,
        torch_package_dir=torch_package_dir,
        windows_root=Path(os.environ.get("SystemRoot", r"C:\Windows")),
    )


@contextmanager
def _cuda_dll_search_path(device_type: str):
    handles: list[Any] = []
    original_path: str | None = None
    try:
        if device_type == "cuda" and os.name == "nt":
            add_directory = getattr(os, "add_dll_directory", None)
            if not callable(add_directory):
                raise ModelLoadError(
                    "Python ne fournit pas os.add_dll_directory(), requis pour "
                    "charger llama.cpp CUDA sous Windows."
                )
            directories = _windows_cuda_dll_directories()
            if not any(
                (directory / "nvcudart_hybrid64.dll").is_file()
                for directory in directories
            ):
                raise ModelLoadError(
                    "Runtime CUDA 13 du pilote NVIDIA introuvable : "
                    "nvcudart_hybrid64.dll est absent du DriverStore Windows. "
                    "Installez le dernier pilote NVIDIA, puis relancez "
                    "install_windows.bat."
                )
            original_path = os.environ.get("PATH", "")
            os.environ["PATH"] = _prepend_dll_directories_to_path(
                original_path,
                directories,
                separator=os.pathsep,
            )
            for directory in directories:
                handles.append(add_directory(str(directory)))
        yield
    finally:
        for handle in reversed(handles):
            handle.close()
        if original_path is not None:
            os.environ["PATH"] = original_path


def load_llama_cpp_embedding_model(
    config: TextEmbeddingConfig,
    *,
    device: str = "cpu",
) -> Any:
    """Charge un GGUF local avec llama.cpp, sans accès réseau."""

    checkpoint = _require_checkpoint(config.checkpoint)
    device_type = _embedding_device_type(device)
    uses_accelerator = device_type in {"cuda", "mps"}
    llama_options: dict[str, Any] = {
        "model_path": str(checkpoint),
        "embedding": True,
        "n_gpu_layers": -1 if uses_accelerator else 0,
        "n_ctx": config.context_size,
        "n_batch": config.context_size,
        # Un encodeur bidirectionnel ne peut pas découper une séquence en
        # micro-lots : llama.cpp exige n_ubatch >= nombre de tokens.
        "n_ubatch": config.batch_size,
        "offload_kqv": uses_accelerator,
        "op_offload": uses_accelerator,
        "flash_attn": uses_accelerator,
        "use_mmap": True,
        "verbose": False,
    }
    if config.threads > 0:
        llama_options["n_threads"] = config.threads
        llama_options["n_threads_batch"] = config.threads

    try:
        with _cuda_dll_search_path(device_type):
            import llama_cpp

            if uses_accelerator:
                info = llama_cpp.llama_print_system_info().decode()
                backend = "MTL" if device_type == "mps" else "CUDA"
                if not llama_cpp.llama_supports_gpu_offload() or backend not in info:
                    raise ModelLoadError(
                        f"Backend {backend} absent de llama-cpp-python pour {checkpoint}. "
                        "Relancez l'installateur PuLID pour restaurer la wheel GPU. "
                        "Aucun repli CPU automatique n'est effectué."
                    )
            return llama_cpp.Llama(
                **llama_options,
            )
    except ImportError as exc:
        raise ModelLoadError(
            "Runtime GGUF absent. Installez l'extra Python 'embeddings' "
            "puis relancez le serveur."
        ) from exc
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(
            f"Impossible de charger le modèle d'embedding GGUF sur {device_type.upper()} : "
            f"{checkpoint}. Vérifiez le fichier et le backend de llama-cpp-python."
        ) from exc


def _require_checkpoint(checkpoint: Path) -> Path:
    resolved = checkpoint.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ModelNotFoundError(
            f"Modèle d'embedding GGUF introuvable : {resolved}. "
            "Placez bge-m3-Q8_0.gguf dans le dossier text_embedding de models_root."
        )
    if resolved.suffix.casefold() != ".gguf":
        raise ModelNotFoundError(
            f"Le modèle d'embedding doit être un fichier .gguf : {resolved}."
        )
    return resolved


def _normalize_inputs(value: str | Sequence[str]) -> list[str]:
    inputs = [value] if isinstance(value, str) else list(value)
    if not inputs:
        raise ValueError("Le champ 'input' doit contenir au moins un texte.")
    if len(inputs) > MAX_EMBEDDING_INPUTS:
        raise ValueError(
            f"Le champ 'input' accepte au maximum {MAX_EMBEDDING_INPUTS} textes."
        )

    selected: list[str] = []
    total_chars = 0
    for index, text in enumerate(inputs):
        if not isinstance(text, str):
            raise ValueError(f"input[{index}] doit être une chaîne de caractères.")
        if not text.strip():
            raise ValueError(f"input[{index}] ne peut pas être vide.")
        if len(text) > MAX_EMBEDDING_INPUT_CHARS:
            raise ValueError(
                f"input[{index}] dépasse la limite de "
                f"{MAX_EMBEDDING_INPUT_CHARS} caractères."
            )
        total_chars += len(text)
        selected.append(text)

    if total_chars > MAX_EMBEDDING_TOTAL_CHARS:
        raise ValueError(
            "La taille cumulée de 'input' dépasse la limite de "
            f"{MAX_EMBEDDING_TOTAL_CHARS} caractères."
        )
    return selected


def _finite_vector(value: Any, *, index: int) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise EmbeddingError(
            f"Le moteur GGUF a renvoyé un vecteur invalide à l'index {index}."
        )

    vector: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise EmbeddingError(
                f"Le vecteur {index} contient une composante non numérique."
            )
        number = float(component)
        if not math.isfinite(number):
            raise EmbeddingError(
                f"Le vecteur {index} contient une composante non finie."
            )
        vector.append(number)

    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        raise EmbeddingError(
            f"Le moteur GGUF a renvoyé un vecteur nul à l'index {index}."
        )
    return [component / norm for component in vector]


def _normalized_response(
    raw: Any,
    *,
    model_id: str,
    input_count: int,
    expected_dimensions: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise EmbeddingError("Réponse invalide du moteur GGUF : objet JSON attendu.")
    raw_data = raw.get("data")
    if not isinstance(raw_data, Sequence) or isinstance(raw_data, (str, bytes)):
        raise EmbeddingError("Réponse invalide du moteur GGUF : champ 'data' absent.")
    if len(raw_data) != input_count:
        raise EmbeddingError(
            f"Le moteur GGUF a renvoyé {len(raw_data)} vecteur(s) pour "
            f"{input_count} entrée(s)."
        )

    data: list[dict[str, Any]] = []
    dimensions: int | None = None
    for index, item in enumerate(raw_data):
        if not isinstance(item, Mapping):
            raise EmbeddingError(
                f"Réponse invalide du moteur GGUF à l'index {index}."
            )
        vector = _finite_vector(item.get("embedding"), index=index)
        if len(vector) != expected_dimensions:
            raise EmbeddingError(
                f"Le vecteur {index} contient {len(vector)} composantes au lieu "
                f"des {expected_dimensions} attendues. Vérifiez le GGUF configuré."
            )
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            raise EmbeddingError("Les vecteurs GGUF n'ont pas tous la même dimension.")
        data.append(
            {
                "object": "embedding",
                "index": index,
                "embedding": vector,
            }
        )

    raw_usage = raw.get("usage")
    usage = raw_usage if isinstance(raw_usage, Mapping) else None
    return {
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": usage,
    }


class TextEmbeddingService:
    """Conserve un unique modèle GGUF paresseux et sérialise son utilisation."""

    def __init__(
        self,
        config: TextEmbeddingConfig | None,
        *,
        device: str = "cpu",
        model_factory: Callable[..., Any] = load_llama_cpp_embedding_model,
    ) -> None:
        self.config = config
        self.device = _embedding_device_type(device)
        self.model_factory = model_factory
        self._model: Any | None = None
        self._lock = Lock()

    def _require_config(self) -> TextEmbeddingConfig:
        if self.config is None:
            raise ModelNotFoundError(
                "Le service d'embedding de texte n'est pas configuré. "
                "Ajoutez la section text_embedding dans config/default.yaml."
            )
        return self.config

    def list_models(self) -> list[dict[str, Any]]:
        config = self._require_config()
        _require_checkpoint(config.checkpoint)
        return [
            {
                "id": config.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "pulid-local",
            }
        ]

    def _get_model(self, config: TextEmbeddingConfig) -> Any:
        if self._model is None:
            self._model = self.model_factory(config, device=self.device)
        return self._model

    @property
    def uses_accelerator(self) -> bool:
        return self.device in {"cuda", "mps"}

    def create_embedding(
        self,
        *,
        model: str,
        input_value: str | Sequence[str],
    ) -> dict[str, Any]:
        config = self._require_config()
        requested_model = model.strip()
        if requested_model != config.model_id:
            raise ValueError(
                f"Modèle d'embedding inconnu : {model!r}. "
                f"Valeur acceptée : {config.model_id!r}."
            )
        inputs = _normalize_inputs(input_value)

        with self._lock:
            engine = self._get_model(config)
            try:
                tokenize = getattr(engine, "tokenize", None)
                if callable(tokenize):
                    for index, text in enumerate(inputs):
                        token_count = len(tokenize(text.encode("utf-8")))
                        if token_count > config.context_size:
                            raise ValueError(
                                f"input[{index}] produit {token_count} jetons pour une "
                                f"fenêtre maximale de {config.context_size}."
                            )
                raw = engine.create_embedding(inputs)
            except ValueError as exc:
                raise ValueError(
                    "Le moteur GGUF a refusé l'entrée. Réduisez sa longueur ; "
                    f"la fenêtre configurée est de {config.context_size} jetons."
                ) from exc
            except Exception as exc:
                raise EmbeddingError(
                    f"Échec du calcul d'embedding avec {config.model_id}."
                ) from exc

        return _normalized_response(
            raw,
            model_id=config.model_id,
            input_count=len(inputs),
            expected_dimensions=config.dimensions,
        )

    def close(self) -> None:
        """Libère le modèle GGUF chargé ; l'appel est idempotent."""

        with self._lock:
            model = self._model
            self._model = None
            if model is None:
                return
            close = getattr(model, "close", None)
            if callable(close):
                close()
