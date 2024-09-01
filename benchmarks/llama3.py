import time

import torch

from vqllm.models.llama3 import llama3_8b


def test_speed():
    bsz, seq_len, dim = 1, 256, 4096
    # bsz, seq_len, dim = 1, 1024, 4096
    # bsz, seq_len, dim = 1, 1, 4

    # quantizer parameters
    num_residual_quantizers = 8
    num_codebooks = 1
    num_codebook_entries = 2048
    codebook_entry_dim = 32
    num_residual_steps = 1
    decay = 0.98

    x = torch.randint(0, 10000, (bsz, seq_len)).cuda()

    llama3 = llama3_8b(
        vq_attn=False,
        vq_layers=[],
        num_codebooks=None,
        num_codebook_entries=None,
        codebook_entry_dim=None,
        num_residual_codebooks=1,
        num_residual_steps=1,
        ema_decay=0.0,
        use_fast_quantizer=False,
    ).cuda()

    with torch.no_grad():
        llama3(x)

    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            llama3(x)
    torch.cuda.synchronize()
    end_time = time.time()
    print("llama3:", end_time - start_time)

    del llama3

    # slow quantizer
    vq_llama3 = llama3_8b(
        vq_attn_key=True,
        vq_attn_value=True,
        vq_layers=[],
        num_codebooks=num_codebooks,
        num_codebook_entries=num_codebook_entries,
        codebook_entry_dim=codebook_entry_dim,
        num_residual_codebooks=num_residual_quantizers,
        num_residual_steps=num_residual_steps,
        ema_decay=0.98,
        use_fast_quantizer=False,
    ).cuda()

    # turn off kmeans init
    for layer in vq_llama3.layers:
        if layer.mlp.quantizer:
            for _, quantizer in layer.mlp.quantizer.items():
                quantizer.codebook.data_initialized = torch.ones(1)
        if layer.attn.quantizer:
            for _, quantizer in layer.attn.quantizer.items():
                quantizer.codebook.data_initialized = torch.ones(1)

    with torch.no_grad():
        vq_llama3(x)

    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            vq_llama3(x)
    torch.cuda.synchronize()
    end_time = time.time()
    print("vq_llama3:", end_time - start_time)

    del vq_llama3

    # fast quantizer
    vq_llama3 = llama3_8b(
        vq_attn_key=True,
        vq_attn_value=True,
        vq_layers=[],
        num_codebooks=num_codebooks,
        num_codebook_entries=num_codebook_entries,
        codebook_entry_dim=codebook_entry_dim,
        num_residual_codebooks=num_residual_quantizers,
        num_residual_steps=num_residual_steps,
        ema_decay=0.98,
        use_fast_quantizer=True,
    ).cuda()

    # turn off kmeans init
    for layer in vq_llama3.layers:
        if layer.mlp.quantizer:
            for _, quantizer in layer.mlp.quantizer.items():
                quantizer.codebook.data_initialized = torch.ones(1)
        if layer.attn.quantizer:
            for _, quantizer in layer.attn.quantizer.items():
                quantizer.codebook.data_initialized = torch.ones(1)

    with torch.no_grad():
        vq_llama3(x)

    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            vq_llama3(x)
    torch.cuda.synchronize()
    end_time = time.time()
    print("vq_llama3:", end_time - start_time)


if __name__ == "__main__":
    test_speed()
