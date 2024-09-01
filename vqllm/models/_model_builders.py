# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtune.modules import TransformerDecoder
from torchtune.modules.tokenizers import SentencePieceTokenizer

from ._component_builders import llama3, mistral


"""
Model builders build specific instantiations using component builders. For example
the llama3_8b model builder uses the llama3 component builder to create the
Llama3 8B model.
"""


def llama3_8b(
    vq_attn_key=False,
    vq_attn_value=False,
    vq_layers=[],
    num_codebooks=None,
    num_codebook_entries=None,
    codebook_entry_dim=None,
    num_residual_codebooks=1,
    num_residual_steps=1,
    ema_decay=0.0,
    use_fast_quantizer=False,
    vq_attn_key_reorder_channel=False,
) -> TransformerDecoder:
    embed_dim = 4096

    if num_codebook_entries is None and num_codebooks is not None:
        num_codebook_entries = embed_dim // num_codebooks
    if codebook_entry_dim is None and num_codebooks is not None:
        codebook_entry_dim = embed_dim // num_codebooks

    if (vq_attn_key or vq_attn_value) and (
        (num_codebooks != 1 and num_codebooks * codebook_entry_dim != embed_dim)
        or (embed_dim % codebook_entry_dim != 0)
    ):
        raise ValueError

    return llama3(
        vocab_size=128_256,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=embed_dim,
        max_seq_len=8192,
        intermediate_dim=14336,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500000.0,
        # vq parameters
        vq_attn_key=vq_attn_key,
        vq_attn_value=vq_attn_value,
        vq_layers=vq_layers,
        num_codebooks=num_codebooks,
        num_codebook_entries=num_codebook_entries,
        codebook_entry_dim=codebook_entry_dim,
        num_residual_codebooks=num_residual_codebooks,
        num_residual_steps=num_residual_steps,
        ema_decay=ema_decay,
        use_fast_quantizer=use_fast_quantizer,
        vq_attn_key_reorder_channel=vq_attn_key_reorder_channel,
    )

def mistral_7b() -> TransformerDecoder:
    """
    Builder for creating a Mistral 7B model initialized w/ the default 7b parameter values
    from https://mistral.ai/news/announcing-mistral-7b/


    Returns:
        TransformerDecoder: Instantiation of Mistral 7B model
    """
    return mistral(
        vocab_size=32_000,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=4096,
        intermediate_dim=14336,
        max_seq_len=32768,
        attn_dropout=0.0,
        norm_eps=1e-5,
    )

def mistral_tokenizer(path: str) -> SentencePieceTokenizer:
    tokenizer = SentencePieceTokenizer(path)
    # Original tokenizer has no pad_id, which causes indexing errors when batch training
    tokenizer.pad_id = 0
    return tokenizer