"""Budget CUDA simulé : aucun GPU, checkpoint ou accès réseau requis."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from pulid_app.exceptions import GenerationError
from pulid_app.models.krea2_memory import Krea2Residency, module_bytes


GIB = 1024**3


@pytest.fixture
def residency():
    class Memory:
        total = 12 * GIB
        external = GIB
        cached = 0
        cleared = 0
        def memory_allocated(self, device):
            return sum(module_bytes(m) for m in models.values() if m.weight.device == device)
        def memory_reserved(self, device):
            return self.memory_allocated(device) + self.cached
        def mem_get_info(self, device):
            return self.total - self.external - self.memory_reserved(device), self.total
        def empty_cache(self):
            self.cached = 0
            self.cleared += 1
        def device(self, device):
            assert device == "cuda:0"
            return nullcontext()

    class Model:
        def __init__(self, size):
            self.weight = SimpleNamespace(device="cpu", numel=lambda: size, element_size=lambda: 1)
            self.transfers = []
        def parameters(self):
            return iter((self.weight,))
        def buffers(self):
            return iter(())
        def to(self, device):
            if device != self.weight.device:
                if device == "cpu":
                    memory.cached += module_bytes(self)
                self.transfers.append(device)
                self.weight.device = device
            return self

    models = {"text_encoder": Model(4 * GIB), "transformer": Model(8 * GIB), "vae": Model(GIB // 2)}
    memory = Memory()
    manager = Krea2Residency(models, device="cuda:0", torch=SimpleNamespace(cuda=memory),
                             reserves={name: 2 * GIB for name in models})
    return manager, memory


def activate(manager, name):
    manager.prepare(name)
    manager.models[name].to(manager.device)


def test_12gb_keeps_transformer_with_vae_between_images(residency):
    manager, memory = residency
    for name in ("text_encoder", "transformer", "vae"):
        activate(manager, name)
    assert not manager.resident("text_encoder")
    assert manager.resident("transformer") and manager.resident("vae")
    assert memory.cleared == 1  # Éviction Qwen seulement.
    manager.begin_generation()
    # Prompt en cache : le débruiteur revient directement, sans aucun transfert.
    for _ in range(3):
        activate(manager, "transformer")
    activate(manager, "vae")
    assert manager.models["transformer"].transfers == ["cuda:0"]
    assert manager.models["vae"].transfers == ["cuda:0"]
    assert memory.cleared == 1


def test_new_prompt_evicts_only_when_qwen_needs_space(residency):
    manager, memory = residency
    activate(manager, "transformer")
    activate(manager, "vae")
    manager.begin_generation()
    assert manager.resident("transformer") and manager.resident("vae")
    activate(manager, "text_encoder")
    assert manager.resident("text_encoder")
    assert not manager.resident("transformer")
    assert manager.resident("vae")  # Évincer Krea suffit, inutile de déplacer le VAE.
    activate(manager, "transformer")
    activate(manager, "vae")
    assert manager.resident("transformer") and manager.resident("vae")
    assert memory.cleared == 2
    assert manager.models["vae"].transfers == ["cuda:0"]


def test_large_gpu_retains_all_components(residency):
    manager, memory = residency
    memory.total = 24 * GIB
    for _ in range(2):
        manager.begin_generation()
        for name in manager.models:
            activate(manager, name)
    assert all(manager.resident(name) for name in manager.models)
    assert all(model.transfers == ["cuda:0"] for model in manager.models.values())
    assert memory.cleared == 0


def test_cached_allocator_space_is_reusable_without_eviction(residency):
    manager, memory = residency
    memory.cached = 7 * GIB
    assert memory.mem_get_info(manager.device)[0] == 4 * GIB
    manager.prepare("transformer")  # 8 Gio de poids + 2 Gio de marge.
    assert memory.cleared == 0


def test_pressure_rechecked_for_resident_model_on_next_generation(residency):
    manager, memory = residency
    activate(manager, "transformer")
    activate(manager, "vae")
    memory.external += GIB
    manager.begin_generation()
    activate(manager, "transformer")
    assert manager.resident("transformer") and not manager.resident("vae")
    assert memory.cleared == 1


def test_budget_failure_is_explicit_and_never_evicts_active_component(residency):
    manager, memory = residency
    activate(manager, "transformer")
    memory.external = 3 * GIB
    manager.begin_generation()
    with pytest.raises(GenerationError, match="VRAM insuffisante pour transformer sur cuda:0"):
        manager.prepare("transformer")
    assert manager.resident("transformer") and manager.active is None


def test_compact_weights_and_buffers_are_counted(tmp_path):
    from pulid_app.paths import configure_external_model_caches
    configure_external_model_caches(tmp_path)
    import torch
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False)
    module.register_buffer("scale", torch.ones(2, dtype=torch.float32))
    assert module_bytes(module) == 16
