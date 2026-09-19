"""Petit serveur HTTP pour l'inventaire SDXL et la génération en mémoire."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from io import BytesIO
import logging
from pathlib import Path
import secrets
import unicodedata
import re
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from pulid_app import __version__
from pulid_app.api_contract import API_CONTRACT_VERSION, capabilities_payload
from pulid_app.config import AppConfig, load_config
from pulid_app.exceptions import (
    FaceNotDetectedError,
    ModelLoadError,
    ModelNotFoundError,
    MultipleFacesDetectedError,
    PuLIDAppError,
    UnsupportedDeviceError,
    actionable_error,
)
from pulid_app.models.identity_encoder import SUPPORTED_IMAGE_FORMATS
from pulid_app.models.text_embedding import (
    TextEmbeddingService,
    load_llama_cpp_embedding_model,
)
from pulid_app.models.sdxl import (
    NORMAL_SIGMA_SCHEDULE,
    SAMPLING_METHOD_SPECS,
    SUPPORTED_SAMPLING_METHODS,
    SUPPORTED_SIGMA_SCHEDULES,
)
from pulid_app.paths import (
    configure_external_model_caches,
    require_models_root,
    resolve_sdxl_checkpoint,
)
from pulid_app.pipeline.generator import DEFAULT_NEGATIVE_PROMPT, ImageGenerator
from pulid_app.pipeline.krea2 import Krea2Generator, Krea2Parameters


LOGGER = logging.getLogger("uvicorn.error")


DEFAULT_METHOD = "default"
DEFAULT_SIGMAS = NORMAL_SIGMA_SCHEDULE
SIGMA_SCHEDULE_ORDER = tuple(
    name
    for name in ("normal", "karras", "exponential", "beta")
    if name in SUPPORTED_SIGMA_SCHEDULES
)
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
MIN_GENERATION_DIMENSION = 64
MAX_GENERATION_DIMENSION = 2048
DEFAULT_IDENTITY_STRENGTH = 0.8
MAX_REFERENCE_BYTES = 20 * 1024 * 1024
MAX_SEED = 2**63 - 1
EMBEDDING_MEMORY_MODES = frozenset(
    {"concurrent", "serialized", "partial", "full", "cpu"}
)


@dataclass(frozen=True)
class GeneratedPayload:
    content: bytes
    filename: str
    seed: int
    model: str
    method: str
    sigmas: str


class OpenAIEmbeddingRequest(BaseModel):
    """Sous-ensemble textuel de la requête OpenAI compatible."""

    model: str = Field(min_length=1, max_length=255)
    input: str | list[str]
    encoding_format: Literal["float"] = "float"


class Krea2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)
    width: int = Field(default=1248, ge=64, le=2048, multiple_of=16)
    height: int = Field(default=832, ge=64, le=2048, multiple_of=16)
    seed: int = Field(default=0, ge=-1, le=MAX_SEED)
    steps: int = Field(default=10, ge=1, le=200)
    cfg: float = Field(default=1, ge=0, le=30, allow_inf_nan=False)
    sampler: Literal["euler"] = "euler"
    scheduler: Literal["beta"] = "beta"
    denoise: float = Field(default=1, ge=0.01, le=1, allow_inf_nan=False)


class Krea2GenerationService:
    def __init__(self, config: AppConfig, *, generator_factory: Callable[..., Any],
                 device: str | None, dtype_name: str | None, offload_strategy: str | None,
                 now_factory: Callable[[], datetime] | None, random_seed: Callable[[], int] | None) -> None:
        self.config = config
        self.generator_factory = generator_factory
        self.options = dict(device=device, dtype_name=dtype_name, offload_strategy=offload_strategy)
        self.now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self.random_seed = random_seed

    def generate(self, parameters: Krea2Parameters) -> GeneratedPayload:
        parameters = replace(parameters, seed=resolve_generation_seed(parameters.seed, self.random_seed))
        generator = self.generator_factory(self.config, **self.options)
        try:
            image, _metadata = generator.generate(parameters)
            filename = generated_filename("krea2", self.now_factory())
            output = BytesIO()
            image.save(output, format="PNG")
            content = output.getvalue()
            return GeneratedPayload(content, filename, parameters.seed, "krea2", parameters.sampler, parameters.scheduler)
        finally:
            generator.close()


def list_sdxl_models(config: AppConfig) -> list[dict[str, Any]]:
    """Liste les checkpoints du dossier SDXL configuré, sans les charger."""

    directory = config.sdxl.checkpoint.parent
    if not directory.is_dir():
        raise ModelNotFoundError(
            f"Dossier de checkpoints SDXL introuvable : {directory}. "
            "Vérifiez le montage du SSD et la clé sdxl.checkpoint."
        )
    configured = config.sdxl.checkpoint.resolve(strict=False)
    return [
        {
            "name": path.stem,
            "filename": path.name,
            "default": path.resolve(strict=False) == configured,
        }
        for path in sorted(directory.glob("*.safetensors"), key=lambda item: item.name.casefold())
        if path.is_file()
    ]


def list_sampling_methods() -> list[dict[str, Any]]:
    """Expose le scheduler du checkpoint et les remplacements disponibles."""

    methods = [
        {
            "name": DEFAULT_METHOD,
            "label": "Scheduler du checkpoint",
            "default": True,
            "supported_sigma_schedules": [DEFAULT_SIGMAS],
        }
    ]
    methods.extend(
        {
            "name": name,
            "label": spec.label,
            "default": False,
            "supported_sigma_schedules": [
                schedule
                for schedule in SIGMA_SCHEDULE_ORDER
                if schedule in spec.supported_sigma_schedules
            ],
        }
        for name, spec in SAMPLING_METHOD_SPECS.items()
    )
    return methods


def list_sigma_schedules() -> list[dict[str, Any]]:
    """Expose les courbes de sigmas indépendamment des méthodes de sampling."""

    labels = {
        NORMAL_SIGMA_SCHEDULE: "Normal / natif",
        "karras": "Karras",
        "exponential": "Exponentiel",
        "beta": "Beta",
    }
    method_names = [DEFAULT_METHOD, *SAMPLING_METHOD_SPECS]
    schedules: list[dict[str, Any]] = []
    for name in SIGMA_SCHEDULE_ORDER:
        compatible_methods = [
            method
            for method in method_names
            if name in _supported_sigmas_for_http_method(method)
        ]
        schedules.append(
            {
                "name": name,
                "label": labels[name],
                "default": name == DEFAULT_SIGMAS,
                "supported_sampling_methods": compatible_methods,
            }
        )
    return schedules


def _supported_sigmas_for_http_method(method: str) -> frozenset[str]:
    if method == DEFAULT_METHOD:
        return frozenset({DEFAULT_SIGMAS})
    return SAMPLING_METHOD_SPECS[method].supported_sigma_schedules


def resolve_generation_seed(
    requested: int,
    random_seed: Callable[[], int] | None = None,
) -> int:
    """Transforme 0/-1 en seed aléatoire et valide les seeds explicites."""

    if requested in {0, -1}:
        factory = random_seed or (lambda: secrets.randbelow(MAX_SEED) + 1)
        selected = int(factory())
        if selected <= 0 or selected > MAX_SEED:
            raise ValueError(f"La seed aléatoire doit être comprise entre 1 et {MAX_SEED}.")
        return selected
    if requested < -1 or requested > MAX_SEED:
        raise ValueError(
            f"La seed doit valoir -1, 0, ou être comprise entre 1 et {MAX_SEED}."
        )
    return requested


def decode_reference_image(content: bytes) -> Image.Image:
    """Décode une référence entièrement en mémoire et valide son format."""

    if not content:
        raise ValueError("L'image de référence est vide.")
    if len(content) > MAX_REFERENCE_BYTES:
        raise ValueError(
            f"L'image de référence dépasse la limite de {MAX_REFERENCE_BYTES // 1024 // 1024} Mio."
        )
    try:
        with Image.open(BytesIO(content)) as source:
            image_format = (source.format or "").upper()
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                raise ValueError(
                    f"Format {image_format or 'inconnu'} non pris en charge. "
                    f"Formats acceptés : {supported}."
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except ValueError:
        raise
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Image de référence illisible : {exc}") from exc
    return image


def _filename_slug(character: str) -> str:
    normalized = unicodedata.normalize("NFKD", character.strip())
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").casefold()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return slug or "personnage"


def generated_filename(character: str, created_at: datetime) -> str:
    """Construit un nom sûr avec le personnage et l'horodatage UTC."""

    utc = created_at.astimezone(timezone.utc)
    return f"{_filename_slug(character)}_{utc.strftime('%Y%m%dT%H%M%S_%fZ')}.png"


def _sampling_method(value: str) -> str | None:
    normalized = value.strip().casefold()
    if normalized == DEFAULT_METHOD:
        return None
    if normalized not in SUPPORTED_SAMPLING_METHODS:
        accepted = ", ".join([DEFAULT_METHOD, *sorted(SUPPORTED_SAMPLING_METHODS)])
        raise ValueError(
            f"Méthode de sampling inconnue : {value!r}. Valeurs acceptées : {accepted}."
        )
    return normalized


def _sigma_schedule(value: str) -> str | None:
    normalized = value.strip().casefold()
    if normalized not in SUPPORTED_SIGMA_SCHEDULES:
        accepted = ", ".join(SIGMA_SCHEDULE_ORDER)
        raise ValueError(
            f"Courbe de sigmas inconnue : {value!r}. Valeurs acceptées : {accepted}."
        )
    return None if normalized == DEFAULT_SIGMAS else normalized


def _sampling_selection(method: str, sigmas: str) -> tuple[str | None, str | None]:
    sampling_method = _sampling_method(method)
    sigma_schedule = _sigma_schedule(sigmas)
    effective_method = sampling_method or DEFAULT_METHOD
    effective_sigmas = sigma_schedule or DEFAULT_SIGMAS
    supported = _supported_sigmas_for_http_method(effective_method)
    if effective_sigmas not in supported:
        accepted = ", ".join(sorted(supported))
        raise ValueError(
            f"La courbe de sigmas {effective_sigmas!r} est incompatible avec "
            f"la méthode {effective_method!r}. Valeurs acceptées : {accepted}."
        )
    return sampling_method, sigma_schedule


class GenerationService:
    """Orchestre une requête et ne persiste que le petit cache d'identité."""

    def __init__(
        self,
        config: AppConfig,
        *,
        device: str | None = None,
        dtype_name: str | None = None,
        offload_strategy: str | None = None,
        generator_factory: Callable[..., Any] = ImageGenerator,
        now_factory: Callable[[], datetime] | None = None,
        random_seed: Callable[[], int] | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype_name = dtype_name
        self.offload_strategy = offload_strategy
        self.generator_factory = generator_factory
        self.now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self.random_seed = random_seed
        self._cuda_generator: Any | None = None
        self._cuda_checkpoint: Path | None = None

    def _uses_cuda(self, generator: Any) -> bool:
        selected_device = self.device
        if selected_device is None:
            try:
                selected_device = str(generator.device)
            except AttributeError:
                return False
        return selected_device.strip().casefold().split(":", maxsplit=1)[0] == "cuda"

    def _close_cuda_generator(self) -> None:
        generator = self._cuda_generator
        self._cuda_generator = None
        self._cuda_checkpoint = None
        if generator is not None:
            generator.close()

    def _generator_for(
        self,
        request_config: AppConfig,
        checkpoint: Path,
    ) -> tuple[Any, bool]:
        normalized_checkpoint = checkpoint.resolve(strict=False)
        if self._cuda_generator is not None:
            if self._cuda_checkpoint == normalized_checkpoint:
                return self._cuda_generator, True
            self._close_cuda_generator()

        generator = self.generator_factory(
            request_config,
            device=self.device,
            dtype_name=self.dtype_name,
            offload_strategy=self.offload_strategy,
            allow_downloads=False,
        )
        if self._uses_cuda(generator):
            self._cuda_generator = generator
            self._cuda_checkpoint = normalized_checkpoint
            return generator, True
        return generator, False

    def close(self) -> None:
        """Libère le générateur CUDA conservé, notamment à l'arrêt du serveur."""

        self._close_cuda_generator()

    def prepare_for_embedding(self, memory_mode: str) -> bool:
        """Applique à SDXL la politique choisie avant de charger BGE sur GPU."""

        generator = self._cuda_generator
        if generator is None or memory_mode in {
            "concurrent",
            "serialized",
            "cpu",
        }:
            return False
        method_name = {
            "partial": "partial_offload_sdxl_for_embedding",
            "full": "full_offload_sdxl_for_embedding",
        }[memory_mode]
        method = getattr(generator, method_name, None)
        if not callable(method):
            raise ModelLoadError(
                f"Le générateur ne prend pas en charge la politique BGE {memory_mode!r}."
            )
        return bool(method())

    def restore_after_embedding(self, memory_mode: str) -> bool:
        """Restaure l'offload partiel avant la prochaine génération SDXL."""

        generator = self._cuda_generator
        if generator is None or memory_mode != "partial":
            return False
        restore = getattr(generator, "restore_sdxl_after_embedding", None)
        if not callable(restore):
            raise ModelLoadError(
                "Le générateur ne sait pas restaurer l'offload SDXL partiel."
            )
        return bool(restore())

    def generate(
        self,
        *,
        reference_content: bytes,
        character: str,
        prompt: str,
        negative_prompt: str | None,
        clip_skip_2: bool,
        model: str,
        cfg: float,
        steps: int,
        strength: float,
        method: str,
        sigmas: str,
        seed: int,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> GeneratedPayload:
        selected_character = character.strip()
        selected_prompt = prompt.strip()
        selected_negative_prompt = (
            DEFAULT_NEGATIVE_PROMPT
            if negative_prompt is None
            else negative_prompt.strip()
        )
        if not selected_character:
            raise ValueError("Le nom du personnage ne peut pas être vide.")
        if not selected_prompt:
            raise ValueError("Le prompt ne peut pas être vide.")

        checkpoint = resolve_sdxl_checkpoint(self.config, model)
        if not checkpoint.is_file():
            raise ModelNotFoundError(
                f"Checkpoint SDXL introuvable : {checkpoint}. "
                "Choisissez un modèle renvoyé par GET /models."
            )
        sampling_method, sigma_schedule = _sampling_selection(method, sigmas)
        effective_method = sampling_method or DEFAULT_METHOD
        effective_sigmas = sigma_schedule or DEFAULT_SIGMAS
        effective_seed = resolve_generation_seed(seed, self.random_seed)
        reference_image = decode_reference_image(reference_content)
        request_config = replace(
            self.config,
            sdxl=replace(self.config.sdxl, checkpoint=checkpoint),
        )

        generator: Any | None = None
        generator_is_retained = False
        generation_error: Exception | None = None
        try:
            generator, generator_is_retained = self._generator_for(
                request_config,
                checkpoint,
            )
            identity = generator.encode_identity_memory(
                reference_image,
                identity_id=selected_character,
                source_name="<http-upload>",
                use_cache=True,
            )
            generated = generator.generate_in_memory(
                prompt=selected_prompt,
                negative_prompt=selected_negative_prompt,
                clip_skip_2=clip_skip_2,
                identity=identity,
                seed=effective_seed,
                steps=steps,
                guidance_scale=cfg,
                sampling_method=sampling_method,
                sigma_schedule=sigma_schedule,
                width=width,
                height=height,
                identity_strength=strength,
            )
            output = BytesIO()
            generated.image.save(output, format="PNG")
            content = output.getvalue()
        except Exception as exc:
            generation_error = exc
            raise
        finally:
            should_close = generator is not None and (
                not generator_is_retained or generation_error is not None
            )
            if should_close:
                if generator is self._cuda_generator:
                    self._cuda_generator = None
                    self._cuda_checkpoint = None
                try:
                    generator.close()
                except PuLIDAppError:
                    if generation_error is None:
                        raise

        return GeneratedPayload(
            content=content,
            filename=generated_filename(selected_character, self.now_factory()),
            seed=effective_seed,
            model=checkpoint.stem,
            method=effective_method,
            sigmas=effective_sigmas,
        )


def _http_error(exc: BaseException) -> HTTPException:
    label, cause = actionable_error(exc)
    client_errors = (
        FaceNotDetectedError,
        MultipleFacesDetectedError,
        ModelNotFoundError,
        UnsupportedDeviceError,
        ValueError,
    )
    status_code = 422 if isinstance(cause, client_errors) else 500
    return HTTPException(
        status_code=status_code,
        detail={"error": label, "message": str(cause)},
    )


def _embedding_http_error(exc: BaseException) -> HTTPException:
    label, cause = actionable_error(exc)
    if isinstance(cause, (ModelNotFoundError, ModelLoadError)):
        status_code = 503
    elif isinstance(cause, ValueError):
        status_code = 422
    else:
        status_code = 500
    return HTTPException(
        status_code=status_code,
        detail={"error": label, "message": str(cause)},
    )


def _embedding_runtime_device(
    config: AppConfig,
    *,
    server_device: str | None,
    memory_mode: str,
) -> str:
    if memory_mode not in EMBEDDING_MEMORY_MODES:
        supported = ", ".join(sorted(EMBEDDING_MEMORY_MODES))
        raise ValueError(
            f"Mode mémoire BGE inconnu : {memory_mode!r}. Valeurs acceptées : {supported}."
        )
    if memory_mode == "cpu":
        return "cpu"

    selected = (server_device or config.device.preferred).strip().casefold()
    device_type = selected.split(":", maxsplit=1)[0]
    if device_type not in {"cuda", "mps"}:
        raise UnsupportedDeviceError(
            "BGE sur GPU exige un serveur CUDA ou MPS. Utilisez --CPU pour "
            "exécuter les embeddings sur le processeur."
        )
    return device_type


def create_app(
    config_path: str | Path | None = None,
    *,
    device: str | None = None,
    dtype_name: str | None = None,
    offload_strategy: str | None = None,
    generator_factory: Callable[..., Any] = ImageGenerator,
    krea2_generator_factory: Callable[..., Any] = Krea2Generator,
    embedding_model_factory: Callable[..., Any] = load_llama_cpp_embedding_model,
    embedding_memory_mode: str = "concurrent",
    now_factory: Callable[[], datetime] | None = None,
    random_seed: Callable[[], int] | None = None,
    cors_origins: Sequence[str] = (),
) -> FastAPI:
    """Construit l'application et coordonne SDXL avec BGE CPU ou GPU."""

    config = load_config(config_path)
    require_models_root(config.models_root)
    configure_external_model_caches(config.models_root)
    normalized_embedding_mode = embedding_memory_mode.strip().casefold()
    embedding_device = _embedding_runtime_device(
        config,
        server_device=device,
        memory_mode=normalized_embedding_mode,
    )
    concurrent_cuda = (
        normalized_embedding_mode == "concurrent" and embedding_device == "cuda"
    )
    service = GenerationService(
        config,
        device=device,
        dtype_name=dtype_name,
        offload_strategy=offload_strategy,
        generator_factory=generator_factory,
        now_factory=now_factory,
        random_seed=random_seed,
    )
    embedding_service = TextEmbeddingService(
        config.text_embedding,
        device=embedding_device,
        model_factory=embedding_model_factory,
    )
    krea2_service = Krea2GenerationService(
        config, generator_factory=krea2_generator_factory, device=device,
        dtype_name=dtype_name, offload_strategy=offload_strategy,
        now_factory=now_factory, random_seed=random_seed,
    )

    def prepare_generation_for_gpu() -> None:
        if normalized_embedding_mode in {"partial", "full"}:
            embedding_service.close()
            service.restore_after_embedding(normalized_embedding_mode)

    def close_services() -> None:
        try:
            service.close()
        finally:
            embedding_service.close()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(close_services)

    app = FastAPI(
        title="PuLID API",
        version=__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.generation_service = service
    app.state.krea2_service = krea2_service
    app.state.text_embedding_service = embedding_service
    app.state.embedding_memory_mode = normalized_embedding_mode
    app.state.concurrent_cuda = concurrent_cuda
    if concurrent_cuda:
        LOGGER.info(
            "Mode CUDA concurrent actif : SDXL et BGE peuvent calculer "
            "simultanément sur le GPU."
        )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=[
                "Content-Disposition",
                "X-Generation-Seed",
                "X-SDXL-Model",
                "X-Generation-Model",
                "X-Sampling-Method",
                "X-Sigma-Schedule",
            ],
        )
    generation_lock = asyncio.Lock()
    embedding_lock = asyncio.Lock()
    accelerator_lock = asyncio.Lock()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "api_contract_version": API_CONTRACT_VERSION,
        }

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {
            "component": "pulid",
            "version": __version__,
            "api_contract_version": API_CONTRACT_VERSION,
        }

    @app.get("/capabilities")
    async def capabilities() -> dict[str, object]:
        embedding_config = config.text_embedding
        return capabilities_payload(
            application_version=__version__,
            embedding_model=(
                embedding_config.model_id if embedding_config is not None else None
            ),
            embedding_dimensions=(
                embedding_config.dimensions if embedding_config is not None else None
            ),
        )

    @app.get("/models")
    async def models() -> dict[str, Any]:
        try:
            return {
                "models": list_sdxl_models(config),
                "sampling_methods": list_sampling_methods(),
                "sigma_schedules": list_sigma_schedules(),
            }
        except PuLIDAppError as exc:
            raise _http_error(exc) from exc

    @app.get("/v1/models")
    async def embedding_models() -> dict[str, Any]:
        try:
            return {
                "object": "list",
                "data": embedding_service.list_models(),
            }
        except (PuLIDAppError, OSError, ValueError) as exc:
            raise _embedding_http_error(exc) from exc

    @app.post("/v1/embeddings")
    async def create_embeddings(request: OpenAIEmbeddingRequest) -> dict[str, Any]:
        try:
            async with embedding_lock:
                if embedding_service.uses_accelerator:
                    async with (
                        nullcontext() if concurrent_cuda else accelerator_lock
                    ):
                        await run_in_threadpool(
                            service.prepare_for_embedding,
                            normalized_embedding_mode,
                        )
                        return await run_in_threadpool(
                            embedding_service.create_embedding,
                            model=request.model,
                            input_value=request.input,
                        )
                return await run_in_threadpool(
                    embedding_service.create_embedding,
                    model=request.model,
                    input_value=request.input,
                )
        except (PuLIDAppError, OSError, RuntimeError, ValueError) as exc:
            LOGGER.exception(
                "Échec de la requête POST /v1/embeddings (%s)",
                type(exc).__name__,
            )
            raise _embedding_http_error(exc) from exc

    @app.post("/generate", response_class=Response)
    async def generate(
        reference: Annotated[bytes, File(description="Image JPEG, PNG, WebP, BMP ou TIFF")],
        character: Annotated[str, Form(min_length=1, max_length=100)],
        prompt: Annotated[str, Form(min_length=1, max_length=4000)],
        model: Annotated[str, Form(min_length=1, max_length=255)],
        negative_prompt: Annotated[str | None, Form(max_length=4000)] = None,
        clip_skip_2: Annotated[bool, Form()] = False,
        cfg: Annotated[float, Form(ge=0, le=30)] = 7.0,
        steps: Annotated[int, Form(ge=1, le=200)] = 20,
        strength: Annotated[
            float,
            Form(ge=0, allow_inf_nan=False),
        ] = DEFAULT_IDENTITY_STRENGTH,
        method: Annotated[str, Form(min_length=1)] = DEFAULT_METHOD,
        sigmas: Annotated[str, Form(min_length=1)] = DEFAULT_SIGMAS,
        seed: Annotated[int, Form(ge=-1, le=MAX_SEED)] = 0,
        width: Annotated[
            int,
            Form(
                ge=MIN_GENERATION_DIMENSION,
                le=MAX_GENERATION_DIMENSION,
                multiple_of=8,
            ),
        ] = DEFAULT_WIDTH,
        height: Annotated[
            int,
            Form(
                ge=MIN_GENERATION_DIMENSION,
                le=MAX_GENERATION_DIMENSION,
                multiple_of=8,
            ),
        ] = DEFAULT_HEIGHT,
    ) -> Response:
        try:
            async with generation_lock:
                generation_kwargs = {
                    "reference_content": reference,
                    "character": character,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "clip_skip_2": clip_skip_2,
                    "model": model,
                    "cfg": cfg,
                    "steps": steps,
                    "strength": strength,
                    "method": method,
                    "sigmas": sigmas,
                    "seed": seed,
                    "width": width,
                    "height": height,
                }
                if embedding_service.uses_accelerator:
                    async with (
                        nullcontext() if concurrent_cuda else accelerator_lock
                    ):
                        await run_in_threadpool(prepare_generation_for_gpu)
                        payload = await run_in_threadpool(
                            service.generate,
                            **generation_kwargs,
                        )
                else:
                    payload = await run_in_threadpool(
                        service.generate,
                        **generation_kwargs,
                    )
        except (PuLIDAppError, OSError, RuntimeError, ValueError) as exc:
            raise _http_error(exc) from exc

        return Response(
            content=payload.content,
            media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{payload.filename}"',
                "Cache-Control": "no-store",
                "X-Generation-Seed": str(payload.seed),
                "X-SDXL-Model": payload.model,
                "X-Sampling-Method": payload.method,
                "X-Sigma-Schedule": payload.sigmas,
            },
        )

    @app.post("/generate/krea2", response_class=Response)
    async def generate_krea2(request: Annotated[Krea2Request, Form()]) -> Response:
        try:
            parameters = Krea2Parameters(**request.model_dump())
            # Ces verrous couvrent aussi le mode CUDA concurrent : Krea ne peut
            # pas décharger BGE/SDXL pendant qu'une requête les utilise.
            async with generation_lock, embedding_lock, accelerator_lock:
                await run_in_threadpool(close_services)
                payload = await run_in_threadpool(krea2_service.generate, parameters)
        except (PuLIDAppError, OSError, RuntimeError, ValueError) as exc:
            raise _http_error(exc) from exc
        return Response(
            content=payload.content, media_type="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="{payload.filename}"',
                "Cache-Control": "no-store", "X-Generation-Seed": str(payload.seed),
                "X-Generation-Model": "krea2", "X-Sampling-Method": payload.method,
                "X-Sigma-Schedule": payload.sigmas,
            },
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serveur HTTP PuLID local.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--network",
        action="store_true",
        help=(
            "Mode réseau avancé : écoute sur 0.0.0.0 et autorise toutes les "
            "origins CORS. À réserver à un réseau privé de confiance."
        ),
    )
    parser.add_argument("--port", type=int, default=12693)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument(
        "--offload",
        choices=("none", "model_cpu_offload"),
    )
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--partial",
        dest="embedding_memory_mode",
        action="store_const",
        const="partial",
        help="BGE sur GPU ; déplace CLIP et le VAE SDXL sur CPU pendant son usage.",
    )
    embedding_group.add_argument(
        "--full",
        dest="embedding_memory_mode",
        action="store_const",
        const="full",
        help="BGE sur GPU ; décharge complètement SDXL pendant son usage.",
    )
    embedding_group.add_argument(
        "--serialized-cuda",
        dest="embedding_memory_mode",
        action="store_const",
        const="serialized",
        help=(
            "BGE et SDXL sur GPU avec un verrou commun, sans offload."
        ),
    )
    embedding_group.add_argument(
        "--CPU",
        dest="embedding_memory_mode",
        action="store_const",
        const="cpu",
        help="BGE sur CPU sans modifier la résidence mémoire de SDXL.",
    )
    parser.set_defaults(embedding_memory_mode="concurrent")
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="Origin frontend autorisée, répétable (ex. http://localhost:3000).",
    )
    return parser


def resolve_network_settings(
    host: str,
    cors_origins: Sequence[str],
    *,
    network: bool,
) -> tuple[str, tuple[str, ...]]:
    """Applique le raccourci réseau sans modifier les réglages locaux par défaut."""

    if not network:
        return host, tuple(cors_origins)
    selected_origins = tuple(cors_origins)
    if "*" not in selected_origins:
        selected_origins = (*selected_origins, "*")
    return "0.0.0.0", selected_origins


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    host, cors_origins = resolve_network_settings(
        args.host,
        args.cors_origin,
        network=args.network,
    )
    import uvicorn

    uvicorn.run(
        create_app(
            args.config,
            device=args.device,
            dtype_name=args.dtype,
            offload_strategy=args.offload,
            embedding_memory_mode=args.embedding_memory_mode,
            cors_origins=cors_origins,
        ),
        host=host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
