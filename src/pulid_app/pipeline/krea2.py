"""Génération Krea 2 autonome, sans adaptateur ni cache d'identité."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from pulid_app.config import AppConfig
from pulid_app.exceptions import GenerationError, ModelNotFoundError, UnsupportedDeviceError
from pulid_app.models.krea2 import load_krea2_pipeline
from pulid_app.paths import configure_external_model_caches
from pulid_app.pipeline.memory import MemoryManager


KREA2_SAMPLERS = ("euler",)
KREA2_SCHEDULERS = ("beta",)


@dataclass(frozen=True)
class Krea2Parameters:
    prompt: str
    width: int = 1248
    height: int = 832
    seed: int = 0
    steps: int = 10
    cfg: float = 1.0
    sampler: str = "euler"
    scheduler: str = "beta"
    denoise: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 4000:
            raise ValueError("prompt doit contenir de 1 à 4000 caractères non vides.")
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int or not 64 <= value <= 2048 or value % 16:
                raise ValueError(f"{name} doit être un entier multiple de 16 entre 64 et 2048.")
        if type(self.steps) is not int or not 1 <= self.steps <= 200:
            raise ValueError("steps doit être un entier entre 1 et 200.")
        if type(self.seed) is not int or not -1 <= self.seed <= 2**63 - 1:
            raise ValueError("seed doit être un entier entre -1 et 2^63-1.")
        for name, minimum, maximum in (("cfg", 0, 30), ("denoise", 0.01, 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} doit être un nombre fini entre {minimum} et {maximum}.")
        if self.sampler not in KREA2_SAMPLERS:
            raise ValueError(f"sampler Krea 2 inconnu : {self.sampler!r}. Valeur acceptée : euler.")
        if self.scheduler not in KREA2_SCHEDULERS:
            raise ValueError(f"scheduler Krea 2 inconnu : {self.scheduler!r}. Valeur acceptée : beta.")


def beta_sigmas(steps: int, denoise: float) -> list[float]:
    """Grille beta(0.6,0.6) sur les 10000 temps Krea/Flux, avant shift.

    ComfyUI déduplique les indices arrondis, ajoute zéro, puis retient les
    steps derniers intervalles pour denoise<1. Diffusers ajoute lui-même zéro
    et applique exactement une fois le shift exponentiel mu=1.15.
    """
    import numpy as np
    from scipy.stats import beta

    count = steps if denoise > 0.9999 else int(steps / denoise)
    indices = np.rint(beta.ppf(1 - np.arange(count) / count, 0.6, 0.6) * 9999).astype(int)
    distinct = indices[np.concatenate(([True], indices[1:] != indices[:-1]))]
    return ((distinct[-steps:] + 1) / 10000).tolist()


def shifted_sigma(value: float) -> float:
    return math.exp(1.15) / (math.exp(1.15) + (1 / value - 1))


class Krea2Generator:
    def __init__(self, config: AppConfig, *, device: str | None = None, dtype_name: str | None = None, offload_strategy: str | None = None) -> None:
        self.config = config
        self.device = (device or config.device.preferred).strip().lower()
        self.dtype_name = dtype_name or config.device.dtype
        self.offload = offload_strategy or config.device.offload_strategy
        self.pipeline: Any | None = None
        self.memory = MemoryManager(config.models_root, device=self.device)
        self._torch: Any | None = None

    def _load(self) -> Any:
        if self.pipeline is not None:
            return self.pipeline
        if self.config.krea2 is None:
            raise ModelNotFoundError("Section krea2 absente. Relancez pulid-install --krea2-only.")
        configure_external_model_caches(self.config.models_root)
        import torch

        self._torch = torch
        self.memory.bind_torch(torch)
        kind = self.device.split(":")[0]
        if kind == "cuda" and not torch.cuda.is_available():
            raise UnsupportedDeviceError("CUDA indisponible pour Krea 2. Utilisez --device cpu ou mps.")
        if kind == "mps" and not torch.backends.mps.is_available():
            raise UnsupportedDeviceError("MPS indisponible pour Krea 2. Utilisez --device cpu ou cuda.")
        if self.dtype_name not in {"float16", "float32", "bfloat16"}:
            raise ValueError(f"Dtype Krea 2 inconnu : {self.dtype_name}.")
        if self.offload not in {"none", "model_cpu_offload"}:
            raise ValueError(f"Offload Krea 2 inconnu : {self.offload}.")
        if self.offload == "model_cpu_offload" and kind != "cuda":
            raise UnsupportedDeviceError("model_cpu_offload Krea 2 exige CUDA ; utilisez --offload none sur MPS/CPU.")
        self.dtype = torch.float32 if kind == "cpu" else getattr(torch, self.dtype_name)
        self.pipeline = load_krea2_pipeline(self.config.krea2, device=self.device, dtype=self.dtype, offload=self.offload)
        return self.pipeline

    def generate(self, parameters: Krea2Parameters) -> tuple[Any, dict[str, Any]]:
        pipeline = self._load()
        torch = self._torch
        sigmas = beta_sigmas(parameters.steps, parameters.denoise)
        try:
            with torch.inference_mode():
                # Générateur CPU également sur MPS ; pas de seed globale partagée.
                generator = torch.Generator(device="cpu").manual_seed(parameters.seed)
                latents = pipeline.prepare_latents(
                    1, 16, parameters.height, parameters.width, self.dtype,
                    torch.device("cpu"), generator,
                )
                latents *= shifted_sigma(sigmas[0])
                result = pipeline(
                    prompt=parameters.prompt.strip(), negative_prompt="",
                    width=parameters.width, height=parameters.height,
                    num_inference_steps=len(sigmas), sigmas=sigmas,
                    guidance_scale=parameters.cfg - 1, generator=generator,
                    latents=latents, max_sequence_length=512,
                )
            metadata = {
                **asdict(parameters), "effective_steps": len(sigmas), "device": self.device,
                "dtype": str(self.dtype), "engine": "krea2", "identity_transfer": False,
                "checkpoint": str(self.config.krea2.checkpoint),
                "text_encoder": str(self.config.krea2.text_encoder),
                "vae": str(self.config.krea2.vae),
            }
            return result.images[0], metadata
        except Exception as exc:
            raise GenerationError(f"Génération Krea 2 impossible avec {self.config.krea2.checkpoint} : {exc}") from exc

    def close(self) -> None:
        pipeline, self.pipeline = self.pipeline, None
        try:
            if pipeline is not None:
                if self.offload == "model_cpu_offload":
                    pipeline.remove_all_hooks()
                pipeline.to("cpu")
        finally:
            del pipeline
            if self._torch is not None:
                self.memory.cleanup(force=True)
