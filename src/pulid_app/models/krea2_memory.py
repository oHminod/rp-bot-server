"""Résidence CUDA des composants Krea, avec éviction seulement sous pression."""
from __future__ import annotations

import logging
from typing import Any

from pulid_app.exceptions import GenerationError


def module_bytes(module: Any) -> int:
    # Les échelles des poids quantifiés et les buffers font partie du budget.
    return sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers()))


class Krea2Residency:
    def __init__(self, models: dict[str, Any], *, device: Any, torch: Any,
                 reserves: dict[str, int]) -> None:
        self.models = models
        self.device = device
        self.torch = torch
        self.reserves = reserves
        self.sizes = {name: module_bytes(model) for name, model in models.items()}
        self.active: str | None = None

    def resident(self, name: str) -> bool:
        return next(self.models[name].parameters()).device == self.device

    def available_bytes(self) -> int:
        cuda = self.torch.cuda
        free, _ = cuda.mem_get_info(self.device)
        # Le cache de l'allocateur est réutilisable sans empty_cache(). Ne pas
        # compter deux fois les poids et activations encore réellement vivants.
        cached = max(0, cuda.memory_reserved(self.device) - cuda.memory_allocated(self.device))
        return free + cached

    def prepare(self, name: str) -> None:
        if self.active == name:
            return
        required = self.reserves[name] + (0 if self.resident(name) else self.sizes[name])
        available = self.available_bytes()
        victims = []
        # Préserver le débruiteur en priorité, notamment pendant le VAE.
        for candidate in ("text_encoder", "vae", "transformer"):
            if available >= required:
                break
            if candidate != name and self.resident(candidate):
                victims.append(candidate)
                available += self.sizes[candidate]
        # Si Krea doit partir de toute façon pour charger Qwen, ne pas évincer
        # aussi le petit VAE lorsque cela ne sert plus à atteindre le budget.
        for candidate in tuple(victims):
            if available - self.sizes[candidate] >= required:
                victims.remove(candidate)
                available -= self.sizes[candidate]
        for candidate in victims:
            self.models[candidate].to("cpu")
            logging.getLogger("uvicorn.error").info(
                "Krea 2 : %s déplacé sur CPU pour laisser la place à %s.", candidate, name)
        if victims:
            # Seulement lors d'une éviction, jamais systématiquement en fin
            # d'image. Rend aussi les blocs libérés aux bibliothèques CUDA.
            with self.torch.cuda.device(self.device):
                self.torch.cuda.empty_cache()
        available = self.available_bytes()
        if available < required:
            raise GenerationError(
                f"VRAM insuffisante pour {name} sur {self.device} : "
                f"{available / 1024**3:.2f} Gio disponibles, {required / 1024**3:.2f} Gio "
                "requis avec la marge de calcul. Libérez de la VRAM ou réduisez la résolution ; "
                "relancez la génération pour réévaluer l'offload."
            )
        self.active = name

    def begin_generation(self) -> None:
        # Revalider la mémoire disponible, y compris si le prochain prompt est
        # en cache et que le débruiteur est encore le premier composant appelé.
        self.active = None

    def log_resident(self) -> None:
        names = [name for name in self.models if self.resident(name)]
        logging.getLogger("uvicorn.error").info(
            "Krea 2 : composants conservés en VRAM : %s (%.2f Gio de poids).",
            ", ".join(names) or "aucun", sum(self.sizes[name] for name in names) / 1024**3)


def retain_krea2_components(pipeline: Any, *, dtype: Any) -> None:
    """Remplace la chaîne d'éviction systématique des hooks par un budget CUDA.

    L'offload séquentiel reste inchangé quand un composant entier ne tient pas.
    Les vrais hooks Accelerate conservent le routage des entrées et du VAE.decode.
    """
    hooks = getattr(pipeline, "_all_hooks", ())
    if not hooks:
        return
    import torch
    from accelerate.hooks import CpuOffload, UserCpuOffloadHook, add_hook_to_module, remove_hook_from_module
    from pulid_app.models.quantized import QuantizedLinear

    models = {name: getattr(pipeline, name) for name in ("text_encoder", "transformer", "vae")}
    element_size = torch.empty((), dtype=dtype).element_size()
    reserves = {
        name: 2 * 1024**3 + max((layer.in_features * layer.out_features * element_size
                               for layer in model.modules() if isinstance(layer, QuantizedLinear)), default=0)
        for name, model in models.items()
    }
    manager = Krea2Residency(models, device=pipeline._offload_device, torch=torch, reserves=reserves)

    class ResidentOffload(CpuOffload):
        def pre_forward(self, module: Any, *args: Any, **kwargs: Any) -> Any:
            manager.prepare(names[id(module)])
            return super().pre_forward(module, *args, **kwargs)

    names = {id(model): name for name, model in models.items()}
    retained_hooks = []
    for previous in hooks:
        model = previous.model
        remove_hook_from_module(model)
        hook = ResidentOffload(execution_device=manager.device)
        add_hook_to_module(model, hook)
        retained_hooks.append(UserCpuOffloadHook(model, hook))
    pipeline._all_hooks = retained_hooks
    pipeline._krea2_residency = manager
    logging.getLogger("uvicorn.error").info(
        "Krea 2 : conservation CUDA adaptative ; éviction des composants seulement si nécessaire.")
