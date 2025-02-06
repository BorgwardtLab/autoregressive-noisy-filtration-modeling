import torch
from einops import rearrange
from torch import nn

from anfm.models.causal_decoder import CausalTransformerEncoderLayer


class FeedForward(nn.Module):
    def __init__(
        self,
        embed_dim,
        ff_dim,
        filtration_size,
        time_embedding=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        if time_embedding:
            self.time_embedding = nn.Parameter(
                torch.zeros((filtration_size, embed_dim))
            )
            nn.init.normal_(self.time_embedding.data)
        else:
            self.time_embedding = None

    def forward(self, x, filtration_size, generation_step=None):
        if generation_step is None:
            # We are in training mode
            assert x.ndim == 2 and x.size(0) % filtration_size == 0, x.shape
            x = rearrange(x, "(T X) D -> X T D", T=filtration_size)
            if self.time_embedding is not None:
                x = x + self.time_embedding.unsqueeze(0)
            x = self.mlp(x)
            x = rearrange(x, "X T D -> (T X) D")
        else:
            # We are in generation mode
            assert x.ndim == 2 and x.size(1) == self.embed_dim, x.shape
            if self.time_embedding is not None:
                x = x + self.time_embedding[generation_step].unsqueeze(0)
            x = self.mlp(x)
        return x


class TimeMixer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_layers,
        num_heads,
        ff_dim,
        filtration_size,
        time_embedding=False,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layers = nn.ModuleList(
            [
                CausalTransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.register_buffer(
            "causal_mask",
            torch.nn.Transformer.generate_square_subsequent_mask(filtration_size),
            persistent=False,
        )
        if time_embedding:
            self.time_embedding = nn.Parameter(
                torch.zeros((filtration_size, embed_dim))
            )
            nn.init.normal_(self.time_embedding.data)
        else:
            self.time_embedding = None
        self.cache = None  # A KV-cache that is only used during generation

    def forward(self, x, filtration_size, generation_step=None):
        if generation_step is None:
            # We are in teacher-forcing mode and x represents a whole sequence (i.e. filtration)
            if self.cache is not None:
                raise RuntimeError(
                    "Called forward in teacher-forcing mode but KV cache is not emptied. Have you stopped generation prematurely?"
                )
            x = rearrange(x, "(T X) D -> X T D", T=filtration_size)
            if self.time_embedding is not None:
                x = x + self.time_embedding.unsqueeze(0)
            for layer in self.layers:
                x = layer(x, src_mask=self.causal_mask, generating=False)
            x = rearrange(x, "X T D -> (T X) D")
            return x
        else:
            # We are in generation mode
            if self.training:
                raise RuntimeError(
                    "Called forward method in sampling mode, but model is still training"
                )
            # Now, x is of shape (bs, d) and represents the current token (the sequence prefix is omitted)
            assert x.ndim == 2 and x.size(1) == self.embed_dim
            bs = x.shape[0]
            if self.time_embedding is not None:
                x = x + self.time_embedding[generation_step].unsqueeze(0)
            x = x.unsqueeze(1)  # Add singleton sequence dimension

            if self.cache is None:
                self.cache = [
                    torch.zeros((bs, 0, self.embed_dim), device=x.device)
                    for _ in range(len(self.layers))
                ]
            assert all(
                layer_cache.size(1) == generation_step for layer_cache in self.cache
            )
            new_cache = [
                torch.cat([self.cache[0], x], dim=1),
            ]

            out = None
            for idx, layer in enumerate(self.layers):
                out = layer(
                    new_cache[idx], generating=True
                )  # Don't need mask because we are only looking at last token
                assert out.ndim == 3 and out.size(1) == 1
                if idx < len(self.layers) - 1:
                    new_cache.append(torch.cat([self.cache[idx + 1], out], dim=1))

            if generation_step < filtration_size - 1:
                self.cache = new_cache
            else:
                # delete the KV-cache
                self.cache = None
            # We only return the new token, i.e. output is of shape (bs, d)
            return out.squeeze(dim=1)
