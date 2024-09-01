import torch
import torch.nn.functional as F

from scipy.cluster.vq import kmeans2
from torch import nn


# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
class EmbeddingEMA(nn.Module):
    def __init__(self, num_codebook_entires, codebook_entry_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        weight = torch.randn(num_codebook_entires, codebook_entry_dim)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.cluster_size = nn.Parameter(
            torch.zeros(num_codebook_entires).float(), requires_grad=False
        )
        self.embed_avg = nn.Parameter(weight.clone().float(), requires_grad=False)

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.add_(
            new_cluster_size, alpha=(1 - self.decay) / self.decay
        )

    def embed_avg_ema_update(self, new_embed_avg):
        self.embed_avg.data.add_(new_embed_avg, alpha=(1 - self.decay) / self.decay)

    def weight_update(self, num_tokens_per_group, num_groups):
        # update cluster size and embedding average
        self.cluster_size.data.mul_(self.decay)
        self.embed_avg.data.mul_(self.decay)

        cluster_size = self.cluster_size.view(num_groups, num_tokens_per_group)
        n = cluster_size.sum(dim=-1, keepdim=True)
        smoothed_cluster_size = (
            (cluster_size + self.eps) / (n + num_tokens_per_group * self.eps) * n
        ).view(-1)

        # normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        embed_normalized = embed_normalized.clip(-100.0, 100.0)
        self.weight.data.copy_(embed_normalized)


class VQVAEQuantize(nn.Module):
    def __init__(
        self,
        num_codebooks,
        num_codebook_entries,
        codebook_entry_dim,
        num_residual_steps=1,
        decay=0.0,
        epsilon=1e-5,
        use_fast_quantizer=False,
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.num_codebook_entries = num_codebook_entries
        self.codebook_entry_dim = codebook_entry_dim
        self.num_residual_steps = num_residual_steps

        # quantizer kernel
        self.use_fast_quantizer = use_fast_quantizer
        if self.use_fast_quantizer and self.num_codebooks > 1:
            raise ValueError("Quantizer kernel only supports shared codebook")

        self.codebook = EmbeddingEMA(
            num_codebooks * num_codebook_entries,
            codebook_entry_dim,
            decay=decay,
            eps=epsilon,
        )

        self.register_buffer("data_initialized", torch.zeros(1))

    def update_codebook(self, z, indices, mask=None):
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(
                    indices, self.num_codebooks * self.num_codebook_entries
                ).type(z.dtype)

                if mask is not None:
                    encodings = encodings * mask.unsqueeze(-1).to(encodings.dtype)

                # EMA cluster size
                encodings_sum = encodings.sum(0)
                self.codebook.cluster_size_ema_update(encodings_sum)

                # EMA embedding average
                embed_sum = encodings.transpose(0, 1) @ z.view(
                    -1, self.codebook_entry_dim
                )
                self.codebook.embed_avg_ema_update(embed_sum)

    def forward(self, z, mask=None, copy_grad=True, cosine_distance=False):
        """
        Args:
            z: tensor of shape (bsz, seq_len, *, dim)
                where * is any number of additional dimensions (e.g. num_heads)
            mask: tensor of shape (bsz, seq_len) or None
            copy_grad: bool

        Returns:
            z_q: tensor of shape (bsz, seq_len, *, dim)
                where * is any number of additional dimensions (e.g. num_heads)
            indices: tensor of shape (bsz, seq_len, *, num_residual_steps * dim // codebook_entry_dim)
                where * is any number of additional dimensions (e.g. num_heads)
            commitment_loss: tensor of shape (bsz, seq_len)
        """
        z_q, indices = self.decode(z, cosine_distance)

        if self.training:
            if mask is not None:
                mask_shape = list(mask.shape) + [1] * (
                    len(indices.shape) - len(mask.shape)
                )
                mask = mask.view(*mask_shape).expand_as(indices)
                mask = mask.reshape(-1)

            self.update_codebook(
                z.view(-1, self.codebook_entry_dim), indices.view(-1), mask
            )

            commitment_loss = F.mse_loss(z, z_q.detach().to(z.dtype), reduction="none")
            commitment_loss = commitment_loss.view(*z.shape[:2], -1).mean(dim=-1)
        else:
            commitment_loss = torch.tensor([0], dtype=z.dtype, device=z.device)

        # noop in forward pass, straight-through gradient estimator in backward pass
        # this is set to false for residual quantizers, where we copy grad at the end manually
        if copy_grad:
            z_q = z + (z_q - z).detach()
        return z_q, indices, commitment_loss

    def decode(self, z, cosine_distance):
        # z: ... x d -> N x dhat x -1
        # this should work even if N is 1, i.e. codebook is shared
        # when codebook is not shared, d should be equal to N*dhat
        input_shape = z.shape
        z = z.view(-1, self.num_codebooks, self.codebook_entry_dim)
        z = z.permute(1, 2, 0)

        # codebook: (N*C) x dhat-> N x C x dhat
        codebook = self.codebook.weight.view(
            self.num_codebooks, self.num_codebook_entries, self.codebook_entry_dim
        )

        # compute nearest code index
        # we do not normalize z here but only the codes
        # theoretically, this is same as when we do normalize
        # but in practice, we have seen different results
        # however, the difference is limited to very few latents
        # latents = F.normalize(z, dim=1)
        if cosine_distance:
            codebook = F.normalize(codebook.float(), dim=2)
            sim = torch.matmul(codebook, z.float())
        else:
            # euclidean distance
            codebook = codebook.float()
            z = z.float()
            sim = -(
                z.pow(2).sum(dim=1, keepdim=True)
                + codebook.pow(2).sum(dim=2, keepdim=True)
                - 2 * torch.matmul(codebook, z)
            )

        indices = sim.argmax(dim=1).permute(1, 0)  # -1 x N

        # add offset for each codebook
        offset = torch.arange(self.num_codebooks, device=codebook.device)
        offset = offset.unsqueeze(0) * self.num_codebook_entries
        indices += offset
        indices = indices.reshape(*input_shape[:-1], -1)

        z_q = self.codebook(indices).view(input_shape)
        return z_q, indices

    def update_codebook_weight(self):
        self.codebook.weight_update(self.num_codebook_entries, self.num_codebooks)

    # https://github.com/karpathy/deep-vector-quantization/blob/main/dvq/model/quantize.py
    def init_codebook(self, z, mask=None, num_samples=100000):
        if self.training and self.data_initialized.item() == 0:
            with torch.no_grad():
                print(
                    "running kmeans!!"
                )  # data driven initialization for the embeddings
                flatten = z.detach()
                if mask is not None:
                    flatten = flatten[mask]
                flatten = flatten.view(-1, self.codebook_entry_dim).float()
                rp = torch.randperm(flatten.size(0))[:num_samples]
                kd = kmeans2(
                    flatten[rp].data.cpu().numpy(),
                    self.num_codebooks * self.num_codebook_entries,
                    minit="points",
                )
                self.codebook.weight.data.copy_(torch.from_numpy(kd[0]))
                self.codebook.embed_avg.data.copy_(torch.from_numpy(kd[0]))
                self.codebook.cluster_size.data.copy_(
                    torch.ones_like(self.codebook.cluster_size)
                )
                self.data_initialized.fill_(1)
