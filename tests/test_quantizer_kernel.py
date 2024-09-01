import copy

import torch
from torchtune import utils

from vqllm.quantizer import fast_quantizer, VQVAEQuantize


def test_input_fp32():
    # input parameters
    bsz, seq_len, dim = 2, 2048, 4096

    # quantizer parameters
    num_residual_quantizers = 8
    num_codebooks = 1
    num_codebook_entries = 2048
    codebook_entry_dim = 32
    num_residual_steps = 1
    decay = 0.98

    # quantization loss weight
    beta = 0.25

    mask = torch.randint(0, 2, size=(bsz, seq_len), dtype=torch.int32).cuda()
    x = torch.randn(bsz, seq_len, dim).cuda()
    x.requires_grad = True

    quantizer = VQVAEQuantize(
        num_codebooks=num_codebooks,
        num_codebook_entries=num_codebook_entries,
        codebook_entry_dim=codebook_entry_dim,
        num_residual_steps=num_residual_steps,
        decay=decay,
        epsilon=1e-5,
        use_fast_quantizer=False,
    ).cuda()
    quantizer.data_initialized = torch.ones(1)

    quantizers = [copy.deepcopy(quantizer) for _ in range(num_residual_quantizers)]
    quantizers_copy = [copy.deepcopy(quantizer) for _ in range(num_residual_quantizers)]

    def slow_quantizer(x, mask=None):
        x_orig = x
        commitment_loss = 0
        for quantizer in quantizers:
            for _ in range(quantizer.num_residual_steps):
                xq_, indices, commitment_loss_ = quantizer(
                    x, mask=mask, copy_grad=False
                )
                commitment_loss += commitment_loss_
                x = x - xq_

        xq = x_orig - x.detach()
        return xq, commitment_loss

    x_copy = x.detach().clone()
    x_copy.requires_grad = True

    x1, loss1 = slow_quantizer(x, mask)
    x2, _, loss2 = fast_quantizer(
        x_copy,
        mask=mask,
        quantizers=quantizers_copy,
        compute_grad=True,
        compute_loss=True,
        update_codebook=True,
        return_indices=False,
    )

    (x1.sum() + beta * (loss1 * mask).mean()).backward()
    (x2.sum() + beta * (loss2 * mask).mean()).backward()

    assert torch.allclose(loss1, loss2)
    assert torch.allclose(x1, x2)
    assert torch.allclose(x.grad, x_copy.grad)

    for quantizer, quantizer_copy in zip(quantizers, quantizers_copy):
        assert torch.allclose(
            quantizer.codebook.embed_avg,
            quantizer_copy.codebook.embed_avg,
            atol=0.000001,
        )
        assert torch.allclose(
            quantizer.codebook.cluster_size, quantizer_copy.codebook.cluster_size
        )


def test_input_bf16():
    # input parameters
    bsz, seq_len, dim = 2, 2048, 4096

    # quantizer parameters
    num_residual_quantizers = 8
    num_codebooks = 1
    num_codebook_entries = 2048
    codebook_entry_dim = 32
    num_residual_steps = 1
    decay = 0.98

    # quantization loss weight
    beta = 0.25

    with utils.set_default_dtype(torch.bfloat16):
        x = torch.randn(bsz, seq_len, dim).cuda()
        x.requires_grad = True
        mask = torch.randint(0, 2, size=(bsz, seq_len), dtype=torch.int32).cuda()

        quantizer = VQVAEQuantize(
            num_codebooks=num_codebooks,
            num_codebook_entries=num_codebook_entries,
            codebook_entry_dim=codebook_entry_dim,
            num_residual_steps=num_residual_steps,
            decay=decay,
            epsilon=1e-5,
            use_fast_quantizer=False,
        ).cuda()
    quantizer.data_initialized = torch.ones(1)

    quantizers = [copy.deepcopy(quantizer) for _ in range(num_residual_quantizers)]
    quantizers_copy = [copy.deepcopy(quantizer) for _ in range(num_residual_quantizers)]

    def slow_quantizer(x, mask=None):
        x_dtype = x.dtype
        x_orig = x.float()
        x = x.float()

        commitment_loss = 0
        for quantizer in quantizers:
            for _ in range(quantizer.num_residual_steps):
                xq_, indices, commitment_loss_ = quantizer(
                    x, mask=mask, copy_grad=False
                )
                commitment_loss += commitment_loss_
                x = x - xq_

        xq = (x_orig - x.detach()).to(x_dtype)
        return xq, commitment_loss.to(x_dtype)

    x_copy = x.detach().clone()
    x_copy.requires_grad = True

    x1, loss1 = slow_quantizer(x, mask)
    x2, indices, loss2 = fast_quantizer(
        x_copy,
        mask=mask,
        quantizers=quantizers_copy,
        compute_grad=True,
        compute_loss=True,
        update_codebook=True,
        return_indices=False,
    )

    (x1.sum() + beta * (loss1 * mask).mean()).backward()
    (x2.sum() + beta * (loss2 * mask).mean()).backward()

    assert torch.allclose(loss1, loss2)
    assert torch.allclose(x1, x2)
    assert torch.allclose(x.grad, x_copy.grad)

    for quantizer, quantizer_copy in zip(quantizers, quantizers_copy):
        assert torch.allclose(
            quantizer.codebook.embed_avg,
            quantizer_copy.codebook.embed_avg,
            atol=0.000001,
        )
        assert torch.allclose(
            quantizer.codebook.cluster_size,
            quantizer_copy.codebook.cluster_size,
        )
