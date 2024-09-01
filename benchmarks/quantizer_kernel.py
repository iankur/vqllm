import copy
import time

import torch
from torchtune import utils
from triton.testing import do_bench

from vqllm.quantizer import fast_quantizer, VQVAEQuantize


def benchmark_input_fp32():
    # input parameters
    bsz, seq_len, dim = 2, 2048, 4096

    # quantizer parameters
    num_residual_quantizers = 8
    num_codebooks = 1
    num_codebook_entries = 2048
    codebook_entry_dim = 32
    num_residual_steps = 1
    decay = 0.98

    with utils.set_default_dtype(torch.float32):
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
                xq_, _, commitment_loss_ = quantizer(x, mask=mask, copy_grad=False)
                commitment_loss += commitment_loss_
                x = x - xq_

        xq = (x_orig - x.detach()).to(x_dtype)
        return xq, commitment_loss.to(x_dtype)

    x_copy = x.detach().clone()
    x_copy.requires_grad = True

    slow_quantizer(x, mask)
    start_time = time.time()
    for _ in range(10):
        slow_quantizer(x, mask)
    torch.cuda.synchronize()
    end_time = time.time()

    print("torch:", end_time - start_time)
    print("torch benchmark:", do_bench(lambda: slow_quantizer(x, mask)))

    fast_quantizer(
        x_copy,
        mask=mask,
        quantizers=quantizers_copy,
        compute_grad=True,
        compute_loss=True,
        update_codebook=True,
        return_indices=False,
    )
    start_time = time.time()
    for _ in range(10):
        fast_quantizer(
            x_copy,
            mask=mask,
            quantizers=quantizers_copy,
            compute_grad=True,
            compute_loss=True,
            update_codebook=True,
            return_indices=False,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    print("triton:", end_time - start_time)
    print(
        "triton benchmark:",
        do_bench(
            lambda: fast_quantizer(
                x_copy,
                mask=mask,
                quantizers=quantizers_copy,
                compute_grad=True,
                compute_loss=True,
                update_codebook=True,
                return_indices=False,
            )
        ),
    )


def benchmark_input_bf16():
    # input parameters
    bsz, seq_len, dim = 2, 2048, 4096

    # quantizer parameters
    num_residual_quantizers = 8
    num_codebooks = 1
    num_codebook_entries = 2048
    codebook_entry_dim = 32
    num_residual_steps = 1
    decay = 0.98

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
                xq_, _, commitment_loss_ = quantizer(x, mask=mask, copy_grad=False)
                commitment_loss += commitment_loss_
                x = x - xq_

        xq = (x_orig - x.detach()).to(x_dtype)
        return xq, commitment_loss.to(x_dtype)

    x_copy = x.detach().clone()
    x_copy.requires_grad = True

    slow_quantizer(x, mask)
    start_time = time.time()
    for _ in range(10):
        slow_quantizer(x, mask)
    torch.cuda.synchronize()
    end_time = time.time()

    print("torch:", end_time - start_time)
    print("torch benchmark:", do_bench(lambda: slow_quantizer(x, mask)))

    fast_quantizer(
        x_copy,
        mask=mask,
        quantizers=quantizers_copy,
        compute_grad=True,
        compute_loss=True,
        update_codebook=True,
        return_indices=False,
    )
    start_time = time.time()
    for _ in range(10):
        fast_quantizer(
            x_copy,
            mask=mask,
            quantizers=quantizers_copy,
            compute_grad=True,
            compute_loss=True,
            update_codebook=True,
            return_indices=False,
        )
    torch.cuda.synchronize()
    end_time = time.time()

    print("triton:", end_time - start_time)
    print(
        "triton benchmark:",
        do_bench(
            lambda: fast_quantizer(
                x_copy,
                mask=mask,
                quantizers=quantizers_copy,
                compute_grad=True,
                compute_loss=True,
                update_codebook=True,
                return_indices=False,
            )
        ),
    )


if __name__ == "__main__":
    benchmark_input_fp32()
    benchmark_input_bf16()
