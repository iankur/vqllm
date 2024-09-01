import torch
from torchtune import utils

from vqllm.quantizer import VQVAEQuantize


def test_ema_dtype():
    dtypes = [torch.bfloat16, torch.float32]

    # quantizer parameters
    num_residual_quantizers = 8
    num_codebooks = 1
    num_codebook_entries = 2048
    codebook_entry_dim = 32
    num_residual_steps = 1
    decay = 0.98

    for dtype in dtypes:
        with utils.set_default_dtype(dtype):
            quantizer = VQVAEQuantize(
                num_codebooks=num_codebooks,
                num_codebook_entries=num_codebook_entries,
                codebook_entry_dim=codebook_entry_dim,
                num_residual_steps=num_residual_steps,
                decay=decay,
                epsilon=1e-5,
                use_fast_quantizer=False,
            ).cuda()

        assert quantizer.codebook.weight.dtype == dtype
        assert quantizer.codebook.cluster_size.dtype == torch.float32
        assert quantizer.codebook.embed_avg.dtype == torch.float32
