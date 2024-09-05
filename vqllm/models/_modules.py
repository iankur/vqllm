# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F

from torch import nn, Tensor

from torchtune.models.gemma.transformer import (
    GemmaTransformerDecoder as OriginalGemmaTransformerDecoder,
)
from torchtune.modules import (
    CausalSelfAttention as OriginalCausalSelfAttention,
    KVCache,
    TransformerDecoder as OriginalTransformerDecoder,
    TransformerDecoderLayer as OriginalTransformerDecoderLayer,
)

from vqllm.quantizer import quantize_key, quantize_value


class CausalSelfAttention(OriginalCausalSelfAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: nn.Module,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            output_proj=output_proj,
            pos_embeddings=pos_embeddings,
            kv_cache=kv_cache,
            max_seq_len=max_seq_len,
            attn_dropout=attn_dropout,
        )
        self.quantizer = nn.ModuleDict({})

    def forward(self, x, vq_mask, mask, input_pos):
        commitment_loss = 0

        # input has shape [b, s, d]
        bsz, seq_len, _ = x.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({seq_len}) of input tensor should be smaller "
                f"than max_seq_len ({self.max_seq_len})"
            )

        # q has shape [b, s, num_heads * head_dim]
        # k has shape [b, s, num_kv_heads * head_dim]
        # v has shape [b, s, num_kv_heads * head_dim]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads

        # q: [b, s, n_kv, q_per_kv, h_d]
        # k: [b, s, n_kv, 1, h_d]
        # v: [b, s, n_kv, 1, h_d]
        q = q.view(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
        k = k.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)
        v = v.view(bsz, seq_len, self.num_kv_heads, 1, self.head_dim)

        # quantize
        key = "key"
        if self.quantizer and key in self.quantizer:
            k, commitment_loss_ = quantize_key(
                k, vq_mask, self.quantizer[key], self.training
            )
            commitment_loss += commitment_loss_

        key = "value"
        if self.quantizer and key in self.quantizer:
            v, commitment_loss_ = quantize_value(
                v, vq_mask, self.quantizer[key], self.training
            )
            commitment_loss += commitment_loss_

        # if needed, expand the key and value tensors to have the same shape
        # as the query tensor by copying values across the relevant dim
        if self.num_heads != self.num_kv_heads:
            k = k.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)
            v = v.expand(bsz, seq_len, self.num_kv_heads, q_per_kv, self.head_dim)

        # llama2 applies the RoPE embeddings on tensors with shape
        # [b, s, n_h, h_d]
        # Reshape the tensors before we apply RoPE
        q = q.reshape(bsz, seq_len, -1, self.head_dim)
        k = k.reshape(bsz, seq_len, -1, self.head_dim)
        v = v.reshape(bsz, seq_len, -1, self.head_dim)

        # Apply positional embeddings
        q = self.pos_embeddings(q, input_pos=input_pos)
        k = self.pos_embeddings(k, input_pos=input_pos)

        # [b, n_h, s, h_d]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Update key-value cache
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # shape: [b, 1, s, s]
        if mask is not None:
            mask = mask[:, None, :, :]

        # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
        output = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attn_dropout,
            is_causal=self.kv_cache is None and mask is None,
        )

        # reshape the output to be the same shape as the input
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.output_proj(output), commitment_loss

    def update_quantizer_weight(self):
        for key in self.quantizer:
            for quantizer in self.quantizer[key]:
                quantizer.update_codebook_weight()


class TransformerDecoderLayer(OriginalTransformerDecoderLayer):
    def forward(
        self,
        x: Tensor,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        vq_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [batch_size x seq_length x embed_dim]
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [batch_size x seq_length x seq_length]. This is applied after
                the query-key multiplication and before the softmax. A value of True in row i
                and column j means token i attends to token j. A value of False means token i
                does not attend to token j. If no mask is specified, a causal mask
                is used by default. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            Tensor: output tensor with same shape as input
                [batch_size x seq_length x embed_dim]

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - d: embed dim

        TODO:
            - Make position of norm configurable
        """
        # Input tensor and attention output have the same shape
        # [b, s, d]
        # Norm applied inside self-attention block
        attn_out, commitment_loss = self.attn(
            self.sa_norm(x),
            mask=mask,
            input_pos=input_pos,
            vq_mask=vq_mask,
        )

        # Residual connection; shape: [b, s, d]
        h = attn_out + x

        # Norm applied inside the feedforward block
        mlp_out = self.mlp(self.mlp_norm(h))

        # Residual connection; shape: [b, s, d]
        out = h + mlp_out
        return out, commitment_loss

    def update_quantizer_weight(self):
        self.attn.update_quantizer_weight()


class TransformerDecoder(OriginalTransformerDecoder):
    def forward(
        self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        return_vq_loss: Optional[bool] = False,
        vq_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape [b x s]
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [b x s x s]. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape [b x s x v]

        Raises:
            ValueError: if causal_mask is set but input_pos is None

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            if mask is not None:
                raise ValueError(
                    "An attention mask was set. Cannot use a non-causal mask for inference"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, input_pos]

        commitment_loss = 0

        for layer in self.layers:
            # shape: [b, s, d]
            h, commitment_loss_ = layer(
                h,
                mask=mask,
                input_pos=input_pos,
                vq_mask=vq_mask,
            )
            commitment_loss = commitment_loss + commitment_loss_

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, out_dim] - out_dim is usually the vocab size
        output = self.output(h).float()
        if return_vq_loss:
            return output, commitment_loss
        return output

    def update_quantizer_weight(self):
        for layer in self.layers:
            layer.update_quantizer_weight()


class GemmaTransformerDecoder(OriginalGemmaTransformerDecoder):
    def forward(
        self,
        tokens: Tensor,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        return_vq_loss: Optional[bool] = False,
        vq_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tokens (Tensor): input tensor with shape [b x s]
            mask (Optional[Tensor]): Optional boolean tensor which contains the attention mask
                with shape [b x s x s]. This is applied after the query-key multiplication and
                before the softmax. A value of True in row i and column j means token i attends
                to token j. A value of False means token i does not attend to token j. If no
                mask is specified, a causal mask is used by default. Default is None.
            input_pos (Optional[Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Note: At the very first step of inference, when the model is provided with a prompt,
        ``input_pos`` would contain the positions of all of the tokens in the prompt
        (eg: ``torch.arange(prompt_length)``). This is because we will need to compute the
        KV values for each position.

        Returns:
            Tensor: output tensor with shape [b x s x v]

        Raises:
            ValueError: if causal_mask is set but input_pos is None

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - v: vocab size
            - d: embed dim
            - m_s: max seq len
        """
        # input tensor of shape [b, s]
        bsz, seq_len = tokens.shape

        # shape: [b, s, d]
        h = self.tok_embeddings(tokens)

        if self.causal_mask is not None:
            if input_pos is None:
                raise ValueError(
                    "Caches are setup, but the position of input token is missing"
                )
            if mask is not None:
                raise ValueError(
                    "An attention mask was set. Cannot use a non-causal mask for inference"
                )
            # shape: [1, input_pos_len, m_s]
            # in most cases input_pos_len should be 1
            mask = self.causal_mask[None, input_pos]

        if self.norm_embeddings:
            hidden_dim = h.size(-1)
            h = h * torch.tensor(hidden_dim**0.5, dtype=h.dtype)

        commitment_loss = 0

        for layer in self.layers:
            # shape: [b, s, d]
            h, commitment_loss_ = layer(
                h,
                mask=mask,
                input_pos=input_pos,
                vq_mask=vq_mask,
            )
            commitment_loss = commitment_loss + commitment_loss_

        # shape: [b, s, d]
        h = self.norm(h)

        # shape: [b, s, v]
        output = F.linear(h, self.tok_embeddings.weight).float()
        if return_vq_loss:
            return output, commitment_loss
        return output
