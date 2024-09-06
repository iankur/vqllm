# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Optional

from torch import nn

from torchtune.models.gemma.rms_norm import GemmaRMSNorm
from torchtune.models.gemma.transformer import GemmaTransformerDecoder as OriginalGemmaTransformerDecoder
from torchtune.models.llama3._model_utils import scale_hidden_dim_for_mlp
from torchtune.modules import (
    FeedForward,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder as OriginalTransformerDecoder,
)

from vqllm.quantizer import VQVAEQuantize
from vqllm.models._modules import CausalSelfAttention, GemmaTransformerDecoder, TransformerDecoder, TransformerDecoderLayer


def llama3(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    rope_base: int = 500000.0,
    intermediate_dim: Optional[int] = None,
    norm_eps: float = 1e-5,
    vq_attn_key: Optional[bool] = False,
    vq_attn_value: Optional[bool] = False,
    vq_layers: Optional[list] = [],
    num_codebooks: Optional[int] = None,
    num_codebook_entries: Optional[int] = None,
    codebook_entry_dim: Optional[int] = None,
    num_residual_codebooks: Optional[int] = 1,
    num_residual_steps: Optional[int] = 1,
    ema_decay: Optional[float] = 0.0,
    use_fast_quantizer: Optional[bool] = False,
    vq_attn_key_reorder_channel: Optional[bool] = False,
) -> OriginalTransformerDecoder:
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads
    rope = RotaryPositionalEmbeddings(
        dim=head_dim, max_seq_len=max_seq_len, base=rope_base
    )
    self_attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )
    hidden_dim = (
        intermediate_dim if intermediate_dim else scale_hidden_dim_for_mlp(embed_dim)
    )
    mlp = FeedForward(
        gate_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
        down_proj=nn.Linear(hidden_dim, embed_dim, bias=False),
        up_proj=nn.Linear(embed_dim, hidden_dim, bias=False),
    )
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    decoder = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

    if vq_attn_key or vq_attn_value:
        quantizer = VQVAEQuantize(
            num_codebooks,
            num_codebook_entries,
            codebook_entry_dim,
            decay=ema_decay,
            num_residual_steps=num_residual_steps,
            use_fast_quantizer=use_fast_quantizer,
        )

    if not vq_layers:
        vq_layers = list(range(num_layers))

    for i, layer in enumerate(decoder.layers):
        if i in vq_layers:
            # key
            if vq_attn_key:
                quantizer_key = copy.deepcopy(quantizer)
                quantizer_key.reorder_channel = vq_attn_key_reorder_channel
                layer.attn.quantizer["key"] = nn.ModuleList(
                    [copy.deepcopy(quantizer_key) for _ in range(num_residual_codebooks)]
                )

            # value
            if vq_attn_value:
                layer.attn.quantizer["value"] = nn.ModuleList(
                    [copy.deepcopy(quantizer) for _ in range(num_residual_codebooks)]
                )

    return decoder


def mistral(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-5,
    rope_base: int = 10_000,
    vq_attn_key: Optional[bool] = False,
    vq_attn_value: Optional[bool] = False,
    vq_layers: Optional[list] = [],
    num_codebooks: Optional[int] = None,
    num_codebook_entries: Optional[int] = None,
    codebook_entry_dim: Optional[int] = None,
    num_residual_codebooks: Optional[int] = 1,
    num_residual_steps: Optional[int] = 1,
    ema_decay: Optional[float] = 0.0,
    use_fast_quantizer: Optional[bool] = False,
    vq_attn_key_reorder_channel: Optional[bool] = False,
) -> OriginalTransformerDecoder:
    """
    Build the decoder assoicated with the mistral model. This includes:
    - Token embeddings
    - num_layers number of TransformerDecoderLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    This does NOT currently include inference-time optimizations such as
    sliding-window attention

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. If specified,
            user should ensure `num_heads` % `num_kv_heads` == 0. Default value is
            `None`, in which case this is the same as MHA
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms
        rope_base (int): base for the rotary positional embeddings. Default: 10_000

    Returns:
        TransformerDecoder: Instantiation of mistral model.
    """
    head_dim = embed_dim // num_heads
    num_kv_heads = num_kv_heads if num_kv_heads else num_heads

    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_attn = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(embed_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )

    mlp = FeedForward(
        gate_proj=nn.Linear(embed_dim, intermediate_dim, bias=False),
        down_proj=nn.Linear(intermediate_dim, embed_dim, bias=False),
        up_proj=nn.Linear(embed_dim, intermediate_dim, bias=False),
    )

    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        sa_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
        mlp_norm=RMSNorm(dim=embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    output_proj = nn.Linear(embed_dim, vocab_size, bias=False)
    decoder = TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=RMSNorm(embed_dim, eps=norm_eps),
        output=output_proj,
    )

    if vq_attn_key or vq_attn_value:
        quantizer = VQVAEQuantize(
            num_codebooks,
            num_codebook_entries,
            codebook_entry_dim,
            decay=ema_decay,
            num_residual_steps=num_residual_steps,
            use_fast_quantizer=use_fast_quantizer,
        )

    if not vq_layers:
        vq_layers = list(range(num_layers))

    for i, layer in enumerate(decoder.layers):
        if i in vq_layers:
            # key
            if vq_attn_key:
                quantizer_key = copy.deepcopy(quantizer)
                quantizer_key.reorder_channel = vq_attn_key_reorder_channel
                layer.attn.quantizer["key"] = nn.ModuleList(
                    [copy.deepcopy(quantizer_key) for _ in range(num_residual_codebooks)]
                )

            # value
            if vq_attn_value:
                layer.attn.quantizer["value"] = nn.ModuleList(
                    [copy.deepcopy(quantizer) for _ in range(num_residual_codebooks)]
                )

    return decoder

def gemma(
    vocab_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int,
    embed_dim: int,
    intermediate_dim: int,
    max_seq_len: int,
    attn_dropout: float = 0.0,
    norm_eps: float = 1e-6,
    rope_base: int = 10_000,
    norm_embeddings: bool = True,
    vq_attn_key: Optional[bool] = False,
    vq_attn_value: Optional[bool] = False,
    vq_layers: Optional[list] = [],
    num_codebooks: Optional[int] = None,
    num_codebook_entries: Optional[int] = None,
    codebook_entry_dim: Optional[int] = None,
    num_residual_codebooks: Optional[int] = 1,
    num_residual_steps: Optional[int] = 1,
    ema_decay: Optional[float] = 0.0,
    use_fast_quantizer: Optional[bool] = False,
    vq_attn_key_reorder_channel: Optional[bool] = False,
) -> OriginalGemmaTransformerDecoder:
    """
    Build the decoder associated with the gemma model. This includes:
    - Token embeddings
    - num_layers number of TransformerDecoderLayer blocks
    - RMS Norm layer applied to the output of the transformer
    - Final projection into token space

    This does NOT currently include inference-time optimizations such as
    sliding-window attention

    Args:
        vocab_size (int): number of tokens in vocabulary.
        num_layers (int): number of layers in the transformer decoder.
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        head_dim (int): dimension of head
        num_kv_heads (int): number of key and value heads.
        embed_dim (int): embedding dimension for self-attention
        intermediate_dim (int): intermediate dimension for MLP
        max_seq_len (int): maximum sequence length the model will be run with,
        attn_dropout (float): dropout value passed onto scaled_dot_product_attention.
            Default: 0.0
        norm_eps (float): epsilon in RMS norms Default: 1e-6
        rope_base (int): base for the rotary positional embeddings. Default: 10_000
        norm_embeddings (bool): whether to apply layer norm before the self-attention
            and mlp layers. Default: True

    Returns:
        GemmaTransformerDecoder: Instantiation of gemma model.
    """
    rope = RotaryPositionalEmbeddings(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
    self_att = CausalSelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_proj=nn.Linear(embed_dim, num_heads * head_dim, bias=False),
        k_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        v_proj=nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False),
        output_proj=nn.Linear(num_heads * head_dim, embed_dim, bias=False),
        pos_embeddings=rope,
        kv_cache=None,
        max_seq_len=max_seq_len,
        attn_dropout=attn_dropout,
    )

    gate_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
    down_proj = nn.Linear(intermediate_dim, embed_dim, bias=False)
    up_proj = nn.Linear(embed_dim, intermediate_dim, bias=False)
    activation = nn.GELU(approximate="tanh")
    mlp = FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj, activation=activation)

    layer = TransformerDecoderLayer(
        attn=self_att,
        mlp=mlp,
        sa_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        mlp_norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
    )
    tok_embeddings = nn.Embedding(vocab_size, embed_dim)
    decoder = GemmaTransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        norm=GemmaRMSNorm(embed_dim, eps=norm_eps),
        norm_embeddings=norm_embeddings,
    )

    if vq_attn_key or vq_attn_value:
        quantizer = VQVAEQuantize(
            num_codebooks,
            num_codebook_entries,
            codebook_entry_dim,
            decay=ema_decay,
            num_residual_steps=num_residual_steps,
            use_fast_quantizer=use_fast_quantizer,
        )

    if not vq_layers:
        vq_layers = list(range(num_layers))

    for i, layer in enumerate(decoder.layers):
        if i in vq_layers:
            # key
            if vq_attn_key:
                quantizer_key = copy.deepcopy(quantizer)
                quantizer_key.reorder_channel = vq_attn_key_reorder_channel
                layer.attn.quantizer["key"] = nn.ModuleList(
                    [copy.deepcopy(quantizer_key) for _ in range(num_residual_codebooks)]
                )

            # value
            if vq_attn_value:
                layer.attn.quantizer["value"] = nn.ModuleList(
                    [copy.deepcopy(quantizer) for _ in range(num_residual_codebooks)]
                )

    return decoder