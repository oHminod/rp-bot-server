"""Longueur dynamique : mêmes tokens, mêmes positions, même diffusion à l'arrondi près."""
from __future__ import annotations

import pytest

from pulid_app.models.krea2 import prompt_sequence_length, workflow_pipeline
from pulid_app.paths import configure_external_model_caches
from pulid_app.pipeline.krea2 import KREA2_TEXT_SEQUENCE_LENGTH, beta_sigmas


@pytest.fixture
def pipeline(tmp_path):
    configure_external_model_caches(tmp_path)
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Transformer2DModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3VLTextConfig, Qwen3VLTextModel

    # Vrai tokenizer local, sans accès réseau ni vocabulaire téléchargé.
    tokenizer = Tokenizer(WordLevel(
        {"[UNK]": 0, "[PAD]": 1, "system": 2, "user": 3, "end": 4, "assistant": 5, "word": 6},
        unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    torch.manual_seed(19)
    text = Qwen3VLTextModel(Qwen3VLTextConfig(hidden_size=16, intermediate_size=32,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8, num_hidden_layers=2, vocab_size=16))
    transformer = Krea2Transformer2DModel(
        num_layers=1, attention_head_dim=8, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=64, timestep_embed_dim=16, text_hidden_dim=16, num_text_layers=2,
        text_num_attention_heads=2, text_num_key_value_heads=2, text_intermediate_size=32,
        num_layerwise_text_blocks=1, num_refiner_text_blocks=1, axes_dims_rope=(2, 2, 4),
    )
    text.eval().requires_grad_(False)
    transformer.eval().requires_grad_(False)
    pipe = workflow_pipeline(transformer=transformer, text_encoder=text, tokenizer=tokenizer,
        vae=None, scheduler=FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True),
        is_distilled=True, text_encoder_select_layers=(1, 2))
    pipe.prompt_template_encode_prefix = "system user "
    pipe.prompt_template_encode_start_idx = 2
    pipe.prompt_template_encode_suffix = "end assistant end assistant end"
    pipe.prompt_template_encode_num_suffix_tokens = 5
    pipe.set_progress_bar_config(disable=True)
    return pipe


@pytest.mark.parametrize("tokens", [0, 1, 100, 507, 508, 600, 1019, 1020, 1300])
def test_effective_length_counts_suffix_and_preserves_1024_ceiling(pipeline, tokens):
    assert KREA2_TEXT_SEQUENCE_LENGTH == 1024
    assert prompt_sequence_length(pipeline, "word " * tokens, KREA2_TEXT_SEQUENCE_LENGTH) == min(tokens + 5, 1024)


@pytest.mark.parametrize("tokens", [0, 7, 600, 1300])
def test_short_encoding_preserves_valid_ids_positions_suffix_and_qwen_features(pipeline, tokens):
    import torch
    prompt = "word " * tokens
    length = prompt_sequence_length(pipeline, prompt, KREA2_TEXT_SEQUENCE_LENGTH)
    seen = []
    def capture(module, args, kwargs):
        seen.append({key: kwargs[key].clone() for key in ("input_ids", "attention_mask", "position_ids")})
    hook = pipeline.text_encoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            padded, padded_mask = pipeline.get_text_hidden_states(prompt, 1024, torch.device("cpu"))
            compact, compact_mask = pipeline.get_text_hidden_states(prompt, length, torch.device("cpu"))
    finally:
        hook.remove()
    before, after = seen
    assert after["input_ids"].shape[1] == length + pipeline.prompt_template_encode_start_idx
    assert after["attention_mask"].all()
    valid = before["attention_mask"][0]
    torch.testing.assert_close(after["input_ids"][0], before["input_ids"][0, valid])
    torch.testing.assert_close(after["position_ids"][:, 0], before["position_ids"][:, 0, valid])
    suffix = pipeline.tokenizer(pipeline.prompt_template_encode_suffix)["input_ids"]
    assert after["input_ids"][0, -5:].tolist() == suffix
    assert compact.shape[1] == length and compact_mask.all()
    torch.testing.assert_close(compact[0], padded[0, padded_mask[0]], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("cfg", [0, .5, 1, 2])
def test_compact_diffusion_matches_padded_reference_with_and_without_cfg(pipeline, cfg):
    import torch
    prompt = "word " * 7
    length = prompt_sequence_length(pipeline, prompt, 1024)
    calls = []
    def capture(module, args, kwargs):
        text_length = kwargs["encoder_hidden_states"].shape[1]
        image_length = kwargs["hidden_states"].shape[1]
        assert kwargs["position_ids"].shape[0] == text_length + image_length
        assert kwargs["encoder_attention_mask"].shape[1] == text_length
        calls.append(text_length)
    kwargs = dict(prompt=prompt, negative_prompt="", width=64, height=64,
                  num_inference_steps=2, sigmas=beta_sigmas(2, 1), guidance_scale=cfg - 1,
                  output_type="latent")
    hook = pipeline.transformer.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.inference_mode():
            before = pipeline(**kwargs, max_sequence_length=1024,
                              generator=torch.Generator().manual_seed(42)).images
            after = pipeline(**kwargs, max_sequence_length=length,
                             generator=torch.Generator().manual_seed(42)).images
    finally:
        hook.remove()
    count = 2 if cfg == 1 else 4
    assert calls == [1024] * count + [length] * count
    torch.testing.assert_close(after, before, atol=3e-6, rtol=3e-5)
