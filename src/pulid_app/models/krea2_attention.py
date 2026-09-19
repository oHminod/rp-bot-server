"""Attention Krea CUDA : éviter les matrices d'attention explicites avec GQA masqué."""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from pulid_app.exceptions import GenerationError

BackendSignature = tuple[SDPBackend, int, int, int]


def needs_kv_expansion(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if query.device.type != "cuda" or query.shape[1] == key.shape[1]:
        return False
    params = torch.backends.cuda.SDPAParams(query, key, value, mask, 0.0, False, True)
    cuda = torch.backends.cuda
    # cuDNN éligible ne signifie pas cuDNN sélectionné : sur Ada, PyTorch 2.13
    # place math avant cuDNN. Déplier K/V ouvre le chemin memory-efficient.
    return not (cuda.flash_sdp_enabled() and cuda.can_use_flash_attention(params))


def select_cuda_backend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        mask: torch.Tensor | None) -> SDPBackend:
    cuda = torch.backends.cuda
    params = cuda.SDPAParams(query, key, value, mask, 0.0, False, query.shape[1] != key.shape[1])
    for backend, enabled, supported in (
        (SDPBackend.FLASH_ATTENTION, cuda.flash_sdp_enabled, cuda.can_use_flash_attention),
        (SDPBackend.EFFICIENT_ATTENTION, cuda.mem_efficient_sdp_enabled, cuda.can_use_efficient_attention),
        (SDPBackend.CUDNN_ATTENTION, cuda.cudnn_sdp_enabled, cuda.can_use_cudnn_attention),
    ):
        if enabled() and supported(params):
            return backend
    raise GenerationError(
        f"Attention Krea CUDA fusionnée indisponible sur {query.device} "
        f"({query.dtype}, Q={tuple(query.shape)}, K={tuple(key.shape)}). "
        "Le repli math est désactivé pour éviter les matrices d'attention volumineuses. "
        "Vérifiez l'environnement avec install_windows.bat --update et relancez "
        "scripts/benchmark_krea2_cuda.py --attention."
    )


def krea2_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               mask: torch.Tensor | None = None, *,
               reported: set[BackendSignature] | None = None) -> torch.Tensor:
    original_kv_heads = key.shape[1]
    if needs_kv_expansion(query, key, value, mask):
        repeats = query.shape[1] // key.shape[1]
        key, value = (tensor.repeat_interleave(repeats, dim=1) for tensor in (key, value))
    kwargs = {"attn_mask": mask, "enable_gqa": query.shape[1] != key.shape[1]}
    if query.device.type != "cuda":
        return F.scaled_dot_product_attention(query, key, value, **kwargs)

    backend = select_cuda_backend(query, key, value, mask)
    signature = (backend, query.shape[1], original_kv_heads, query.shape[2])
    if reported is not None and signature not in reported:
        logging.getLogger("uvicorn.error").info(
            "Krea 2 : attention CUDA %s, séquence %d, têtes Q=%d K/V=%d→%d ; repli math désactivé.",
            backend.name, query.shape[2], query.shape[1], original_kv_heads, key.shape[1],
        )
        reported.add(signature)
    # Un seul backend autorisé : le choix ne dépend plus des priorités globales.
    # Le contexte restaure les réglages précédents, même en cas d'erreur/OOM.
    with sdpa_kernel(backend):
        return F.scaled_dot_product_attention(query, key, value, **kwargs)


class Krea2CudaAttnProcessor:
    """Garde SDPA et les projections Diffusers ; adapte seulement les têtes K/V.

    Répète K/V si nécessaire et impose un noyau fusionné. Le coût mémoire
    reste linéaire en séquence, sans repli silencieux vers le backend math.
    """
    def __init__(self, reported: set[BackendSignature] | None = None) -> None:
        self.reported = reported if reported is not None else set()

    def __call__(self, attn: Any, hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor | None = None,
                 image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        from diffusers.models.embeddings import apply_rotary_emb

        query = attn.norm_q(attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim)))
        key = attn.norm_k(attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim)))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        result = krea2_sdpa(query, key, value, attention_mask, reported=self.reported)
        result = result.transpose(1, 2).flatten(2)
        return attn.to_out[0](result * torch.sigmoid(attn.to_gate(hidden_states)))


def configure_krea2_attention(transformer: Any, device: str) -> None:
    if device.split(":")[0] != "cuda":
        return
    from diffusers.models.transformers.transformer_krea2 import Krea2Attention

    reported: set[BackendSignature] = set()
    for module in transformer.modules():
        if isinstance(module, Krea2Attention):
            module.set_processor(Krea2CudaAttnProcessor(reported))
