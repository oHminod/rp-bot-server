"""Attention Krea CUDA : éviter les matrices d'attention explicites avec GQA masqué."""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def needs_kv_expansion(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None) -> bool:
    if query.device.type != "cuda" or mask is None or query.shape[1] == key.shape[1]:
        return False
    params = torch.backends.cuda.SDPAParams(query, key, value, mask, 0.0, False, True)
    cuda = torch.backends.cuda
    return not (
        (cuda.flash_sdp_enabled() and cuda.can_use_flash_attention(params))
        or (cuda.cudnn_sdp_enabled() and cuda.can_use_cudnn_attention(params))
    )


class Krea2CudaAttnProcessor:
    """Garde SDPA et les projections Diffusers ; adapte seulement les têtes K/V.

    Comme ComfyUI, répète K/V lorsque les noyaux fusionnés ne supportent pas
    ensemble GQA et masque. Le coût est linéaire en séquence, au lieu de
    matérialiser une matrice quadratique par tête via le backend math.
    """
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
        if needs_kv_expansion(query, key, value, attention_mask):
            repeats = query.shape[1] // key.shape[1]
            key, value = (tensor.repeat_interleave(repeats, dim=1) for tensor in (key, value))
        result = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask,
            enable_gqa=query.shape[1] != key.shape[1])
        result = result.transpose(1, 2).flatten(2)
        return attn.to_out[0](result * torch.sigmoid(attn.to_gate(hidden_states)))


def configure_krea2_attention(transformer: Any, device: str) -> None:
    if device.split(":")[0] != "cuda":
        return
    from diffusers.models.transformers.transformer_krea2 import Krea2Attention

    for module in transformer.modules():
        if isinstance(module, Krea2Attention):
            module.set_processor(Krea2CudaAttnProcessor())
