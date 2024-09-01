import torch
import torch.nn.functional as F
import triton
import triton.language as tl


"""
Triton kernel for residual vector quantization
1. Input and codebooks are kept in provided precision, e.g. fp32 or bf16
2. Similarity and residual computation happens in fp32
3. EMA update (only during training) happens in fp32. After each EMA update,
codebook is updated with EMA codebook casted to match codebook's precision
"""

TRITON_PRINT_AUTOTUNING = 1


def is_cuda():
    return True  # triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"


def get_cuda_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 256}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_C": 256}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_C": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 32}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_C": 32}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 32, "BLOCK_SIZE_C": 64}, num_stages=5, num_warps=2
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 32, "BLOCK_SIZE_C": 64}, num_stages=2, num_warps=2
        ),
        # Good config for fp8 inputs.
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 256}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 256, "BLOCK_SIZE_C": 128}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 256, "BLOCK_SIZE_C": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_C": 256}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_C": 128}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 32}, num_stages=4, num_warps=4
        ),
    ]


# TODO test MI200 autotune config
def get_hip_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 256, "waves_per_eu": 2},
            num_warps=4,
            num_stages=0,
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 256, "BLOCK_SIZE_C": 256, "waves_per_eu": 2},
            num_warps=8,
            num_stages=0,
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_C": 128, "waves_per_eu": 2},
            num_warps=8,
            num_stages=0,
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_C": 128, "waves_per_eu": 3},
            num_warps=4,
            num_stages=0,
        ),
        triton.Config(
            {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_C": 64, "waves_per_eu": 8},
            num_warps=4,
            num_stages=0,
        ),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# @triton.autotune(
#     configs=get_autotune_config(),
#     key=["N", "num_residual_quantizers", "num_codebook_entries", "num_residual_steps"],
# )
@triton.jit
def _kernel_fwd(
    x_ptr,
    dx_ptr,
    xq_ptr,
    mask_ptr,
    commitment_loss_ptr,
    codebook_weights_ptr,
    codebook_weights_normed_ptr,
    ema_cluster_avg_ptr,
    ema_cluster_size_ptr,
    ema_update_factor: tl.constexpr,
    indices_ptr,
    x_stride,
    dx_stride,
    xq_stride,
    mask_stride,
    commitment_loss_stride,
    codebook_stride,
    codebook_normed_stride,
    ema_cluster_avg_stride,
    ema_cluster_size_stride,
    indices_stride,
    N: tl.constexpr,  # noqa: N803
    num_residual_quantizers: tl.constexpr,
    num_codebook_entries: tl.constexpr,
    num_residual_steps: tl.constexpr,
    compute_loss: tl.constexpr,
    compute_grad: tl.constexpr,
    update_codebook: tl.constexpr,
    return_indices: tl.constexpr,
    cosine_distance: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, N)
    rows = tl.arange(0, BLOCK_SIZE_X) + pid * BLOCK_SIZE_X
    mask = tl.load(mask_ptr + rows * mask_stride)
    x_row = tl.load(
        x_ptr + rows[:, None] * x_stride + cols[None, :],
    )

    if compute_grad:
        dx = tl.zeros_like(x_row).to(tl.float32)
    if compute_loss:
        commitment_loss = tl.zeros([BLOCK_SIZE_X], dtype=tl.float32)
    if return_indices:
        indices_ptr += pid * BLOCK_SIZE_X

    x = x_row.to(tl.float32)
    codebook_rows = tl.arange(0, BLOCK_SIZE_C)

    # iterate over each residual quantizer
    # each residual quantizer can have multiple steps
    for _ in range(num_residual_quantizers):
        for _ in range(num_residual_steps):
            codebook_normed_ptrs = (
                codebook_weights_normed_ptr
                + codebook_rows[None, :]
                + cols[:, None] * codebook_normed_stride
            )
            cur_score = tl.full([BLOCK_SIZE_X], -float("inf"), dtype=tl.float32)
            cur_idx = tl.full([BLOCK_SIZE_X], -1, dtype=tl.int32)

            # iterate over each block of codebook entries
            for k in range(num_codebook_entries // BLOCK_SIZE_C):
                codebook_normed = tl.load(codebook_normed_ptrs)

                # compute nearest match
                if cosine_distance:
                    score = tl.dot(x, codebook_normed, allow_tf32=False)
                else:
                    # euclidean distance
                    score = -(
                        tl.expand_dims(tl.sum(x * x, axis=1), axis=1)
                        + tl.expand_dims(
                            tl.sum(codebook_normed * codebook_normed, axis=0), axis=0
                        )
                        - 2 * tl.dot(x, codebook_normed, allow_tf32=False)
                    )
                score, idx = tl.max(score, axis=1, return_indices=True)
                idx += k * BLOCK_SIZE_C

                # update best match so far
                cur_idx = tl.where(score > cur_score, idx, cur_idx)
                cur_score = tl.maximum(score, cur_score)

                codebook_normed_ptrs += BLOCK_SIZE_C

            xq = tl.load(
                codebook_weights_ptr
                + cur_idx[:, None] * codebook_stride
                + cols[None, :],
            )

            # update residual, cluster sum and size
            if update_codebook:
                ema_cluster_avg_ptrs = (
                    ema_cluster_avg_ptr
                    + cur_idx[:, None] * ema_cluster_avg_stride
                    + cols[None, :]
                )
                tl.atomic_add(
                    ema_cluster_avg_ptrs,
                    x * tl.expand_dims(mask, 1) * ema_update_factor,
                )

                ema_cluster_size_ptrs = (
                    ema_cluster_size_ptr + cur_idx * ema_cluster_size_stride
                )
                tl.atomic_add(ema_cluster_size_ptrs, mask * ema_update_factor)

            # update residual, quantization loss and gradient as follows:
            # commitment_loss_ += (tl.sum((x  - xq_) ** 2) / N) * mask
            # dx_ += 2 * (x - xq_) / N * mask
            x -= xq  # * tl.expand_dims(mask, 1)

            if compute_grad:
                dx += 2 * x / N
            if compute_loss:
                commitment_loss += tl.sum(x * x, axis=1) / N
            if return_indices:
                tl.store(
                    indices_ptr + rows,
                    cur_idx,
                )
                indices_ptr += indices_stride

        # advance pointer for next residual codebook
        codebook_weights_ptr += codebook_stride * num_codebook_entries
        codebook_weights_normed_ptr += num_codebook_entries

        if update_codebook:
            ema_cluster_avg_ptr += ema_cluster_avg_stride * num_codebook_entries
            ema_cluster_size_ptr += ema_cluster_size_stride * num_codebook_entries

    x = x_row - x
    tl.store(
        xq_ptr + rows[:, None] * xq_stride + cols[None, :],
        x.to(xq_ptr.type.element_ty),
    )

    if compute_grad:
        dx = dx * tl.expand_dims(mask, 1)
        tl.store(
            dx_ptr + rows[:, None] * dx_stride + cols[None, :],
            dx.to(dx_ptr.type.element_ty),
        )

    if compute_loss:
        tl.store(
            commitment_loss_ptr + rows * commitment_loss_stride,
            commitment_loss.to(commitment_loss_ptr.type.element_ty),
        )


class FastQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        mask,
        quantizers,
        compute_loss,
        compute_grad,
        update_codebook,
        return_indices,
        cosine_distance,
    ):
        BLOCK_SIZE_X = 32  # noqa: N806
        BLOCK_SIZE_C = 64  # noqa: N806
        num_warps = 2

        num_residual_quantizers = len(quantizers)
        num_codebook_entries = quantizers[0].num_codebook_entries
        num_residual_steps = quantizers[0].num_residual_steps
        codebook_entry_dim = quantizers[0].codebook_entry_dim

        # NOTE: assumes same quantizer is applied across hidden dim
        x_shape = x.shape
        x = x.view(-1, codebook_entry_dim)
        M, N = x.shape  # noqa: N806

        if M % BLOCK_SIZE_X != 0 or num_codebook_entries % BLOCK_SIZE_C != 0:
            raise ValueError

        if mask is None:
            mask = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        else:
            mask = mask.unsqueeze(-1).expand(-1, -1, M // torch.numel(mask))
            mask = mask.reshape(-1).to(x.dtype)

        xq = torch.empty_like(x)
        codebook_weights = torch.concat([q.codebook.weight for q in quantizers], dim=0)
        # tl.dot requires both arguments to have same type and we cast x to fp32
        if cosine_distance:
            codebook_weights_normed = F.normalize(codebook_weights.float())
        else:
            codebook_weights_normed = codebook_weights.float()
        codebook_weights_normed = codebook_weights_normed.t().contiguous()

        if compute_loss:
            commitment_loss = torch.empty_like(x[:, 0]).float()
            commitment_loss_stride = commitment_loss.stride(0)
        else:
            commitment_loss = 0
            commitment_loss_stride = 0

        if compute_grad:
            dx = torch.empty_like(x)
            dx_stride = dx.stride(0)
        else:
            dx, dx_stride = None, 0

        ema_decay = quantizers[0].codebook.decay
        ema_update_factor = (1 - ema_decay) / ema_decay
        if update_codebook:
            ema_cluster_avg = torch.concat(
                [q.codebook.embed_avg for q in quantizers], dim=0
            )
            ema_cluster_size = torch.concat(
                [q.codebook.cluster_size for q in quantizers], dim=0
            )

            # triton atomic_add does not support some types (e.g. bf16)
            if ema_cluster_avg.dtype != torch.float32:
                raise ValueError
            if ema_cluster_size.dtype != torch.float32:
                raise ValueError

            ema_cluster_avg_stride = ema_cluster_avg.stride(0)
            ema_cluster_size_stride = ema_cluster_size.stride(0)
        else:
            ema_cluster_avg, ema_cluster_size = None, None
            ema_cluster_avg_stride = 0
            ema_cluster_size_stride = 0

        if return_indices:
            indices = torch.empty(
                (num_residual_quantizers * num_residual_steps, x.shape[0]),
                dtype=torch.int32,
                device=x.device,
            )
            indices_stride = indices.stride(0)
        else:
            indices = None
            indices_stride = 0

        _kernel_fwd[(M // BLOCK_SIZE_X,)](
            # _kernel_fwd[lambda META: (triton.cdiv(M, META['BLOCK_SIZE_X']),)](
            x,
            dx,
            xq,
            mask,
            commitment_loss,
            codebook_weights,
            codebook_weights_normed,
            ema_cluster_avg,
            ema_cluster_size,
            ema_update_factor,
            indices,
            x.stride(0),
            dx_stride,
            xq.stride(0),
            mask.stride(0),
            commitment_loss_stride,
            codebook_weights.stride(0),
            codebook_weights_normed.stride(0),
            ema_cluster_avg_stride,
            ema_cluster_size_stride,
            indices_stride,
            N,
            num_residual_quantizers,
            num_codebook_entries,
            num_residual_steps,
            compute_loss,
            compute_grad,
            update_codebook,
            return_indices,
            cosine_distance,
            BLOCK_SIZE_X,
            BLOCK_SIZE_C,
            num_warps=num_warps,
        )

        # update ema cluster size and avg
        if update_codebook:
            ema_cluster_avg = ema_cluster_avg.chunk(num_residual_quantizers)
            ema_cluster_size = ema_cluster_size.chunk(num_residual_quantizers)
            for quantizer, avg, size in zip(
                quantizers, ema_cluster_avg, ema_cluster_size
            ):
                quantizer.codebook.embed_avg.data.copy_(avg)
                quantizer.codebook.cluster_size.data.copy_(size)

        ctx.compute_grad = compute_grad
        if compute_grad:
            dx = dx.view(*x_shape) / (M / (x_shape[0] * x_shape[1]))
            ctx.save_for_backward(dx)

        if compute_loss:
            commitment_loss = commitment_loss.view(*x_shape[:2], -1).mean(dim=-1)
            commitment_loss = commitment_loss.to(x.dtype)
        else:
            commitment_loss = 0

        if return_indices:
            indices = indices.view(
                num_residual_quantizers, num_residual_steps, *x_shape[:-1], -1
            )
            indices = indices.permute(*list(range(len(indices.shape))[2:]), 0, 1)
        else:
            indices = None

        xq = xq.view(*x_shape)
        return xq, indices, commitment_loss

    @staticmethod
    def backward(ctx, d_xq, d_indices, d_commiment_loss):
        if not ctx.compute_grad:
            return d_xq, None, None, None, None, None, None, None

        d_x = ctx.saved_tensors[0]
        if d_commiment_loss is not None:
            d_x = d_x * d_commiment_loss.unsqueeze(-1)
        return d_x + d_xq, None, None, None, None, None, None, None


def fast_quantizer(
    x,
    mask,
    quantizers,
    compute_loss,
    compute_grad=False,
    update_codebook=False,
    return_indices=False,
    cosine_distance=False,
):
    return FastQuantizer.apply(
        x,
        mask,
        quantizers,
        compute_loss,
        compute_grad,
        update_codebook,
        return_indices,
        cosine_distance,
    )
