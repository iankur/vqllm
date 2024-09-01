import torch

from .quantize_kernel import fast_quantizer


def quantize(
    x,
    mask,
    quantizers,
    training=False,
    scale=None,
    cosine_distance=False,
):
    """
    Args:
        x: tensor of shape (bsz, seq_len, *, dim)
            where * is any number of additional dimensions (e.g. num_heads)
        mask: tensor of shape (bsz, seq_len) or None
        quantizers: list of quantizers
        training: bool
        scale: tensor of shape (bsz, seq_len, *, 1) or None
    """
    # Compute scale for each token
    scale = x.std(dim=-1, keepdim=True)
    x = x / scale

    if quantizers[0].use_fast_quantizer and quantizers[0].data_initialized.item() == 1:
        x, _, commitment_loss = fast_quantizer(
            x,
            mask,
            quantizers,
            compute_loss=True,
            update_codebook=training,
            return_indices=False,
            cosine_distance=cosine_distance,
        )
    else:
        x_orig = x.float()
        x_dtype = x.dtype

        with torch.no_grad():
            x = x.float()
            commitment_loss = 0
            for quantizer in quantizers:
                if quantizer.data_initialized.item() == 0:
                    quantizer.init_codebook(x, mask)

                for _ in range(quantizer.num_residual_steps):
                    xq_, indices, commitment_loss_ = quantizer(
                        x, mask=mask, copy_grad=False, cosine_distance=cosine_distance
                    )
                    x = x - xq_
                    commitment_loss += commitment_loss_

        x = (x_orig - x.detach()).to(x_dtype)
        commitment_loss = commitment_loss.to(x_dtype)

    x = x * scale
    return x, commitment_loss


def quantize_key(x, mask, quantizers, training):
    cosine_distance = False
    reorder_channel = quantizers[0].reorder_channel

    if reorder_channel:
        x_shape = x.shape
        dim = x_shape[-1]
        codebook_entry_dim = quantizers[0].codebook_entry_dim

        x = x.view(*x_shape[:-1], -1, dim // codebook_entry_dim)
        x = x.transpose(-1, -2).reshape(*x_shape)

    x, commitment_loss = quantize(x, mask, quantizers, training, cosine_distance)

    if reorder_channel:
        x = x.split(codebook_entry_dim, dim=-1)
        x = torch.stack(x, dim=-1).reshape(*x_shape)

    return x, commitment_loss


def quantize_value(x, mask, quantizers, training):
    cosine_distance = False
    return quantize(x, mask, quantizers, training, cosine_distance)
