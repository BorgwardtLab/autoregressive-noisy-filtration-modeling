import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric import utils

from anfm.models.batch_renorm import BatchRenorm1d


class DiSAT(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        max_time_steps,
        num_edge_classes=1,
        num_heads=8,
        gnn_type="gine",
        k_hop=2,
        use_cls_token=False,
        global_embed_dim=None,
        use_cycle_features=False,
        node_time_embedding=False,
        attn_drop=0.0,
        batch_renorm=False,
    ):
        super().__init__()

        assert not use_cycle_features  # TODO: Remove the cycle feature arg completely

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_edge_classes = num_edge_classes

        self.use_cycle_features = use_cycle_features

        if use_cycle_features:
            self.cycle_embed = nn.Sequential(
                nn.Linear(3, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        if num_edge_classes > 1:
            self.edge_embed = nn.Embedding(num_edge_classes, embed_dim)
        else:
            self.edge_embed = None

        self.t_embed = nn.Embedding(max_time_steps, embed_dim)
        if node_time_embedding:
            # in this case, add time embedding to initial node features
            self.t_embed_nodes = nn.Embedding(max_time_steps, embed_dim)
        else:
            self.t_embed_nodes = None

        self.global_embed_dim = global_embed_dim
        if global_embed_dim is not None:
            self.global_embed = nn.Sequential(
                nn.Linear(global_embed_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    gnn_type=gnn_type,
                    k_hop=k_hop,
                    use_cls_token=use_cls_token,
                    attn_drop=attn_drop,
                    batch_renorm=batch_renorm,
                )
                for _ in range(num_layers)
            ]
        )

        self.gnn_type = gnn_type

        self.initialize_weights()

    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def forward(
        self,
        edge_index,
        batch,
        x=None,
        edge_attr=None,
        pos_encoding=None,
        t=None,
        global_x=None,
    ):
        if x is None:
            x = torch.zeros((batch.shape[0], self.embed_dim), device=edge_index.device)

        if edge_attr is not None:
            assert self.edge_embed is not None
            edge_attr = self.edge_embed(edge_attr.squeeze())

        t_embedding = self.t_embed(t)
        if self.global_embed_dim is not None and global_x is not None:
            t_embedding = t_embedding + self.global_embed(global_x)

        # Convert to dense batch
        x, mask = utils.to_dense_batch(x, batch)

        if self.t_embed_nodes is not None:
            t2 = self.t_embed_nodes(t)
            x = x + t2.unsqueeze(1)

        # SA-Transformer encoder
        for block in self.blocks:
            x, pos_encoding = block(
                x, t_embedding, edge_index, edge_attr, mask, pos_encoding
            )
        result = {"node_embeddings": x, "mask": mask, "pos_encoding": pos_encoding}
        return result

    @classmethod
    def build_model(cls, name, **model_args):
        model = get_model(cls, name, **model_args)
        return model


def get_model(cls, name="small", **model_kwargs):
    if name == "small":
        model = cls(num_layers=6, embed_dim=256, **model_kwargs)
    elif name == "tiny":
        model = cls(num_layers=6, embed_dim=64, **model_kwargs)
    elif name == "deep":
        model = cls(num_layers=12, embed_dim=128, **model_kwargs)
    elif name == "base":
        model = cls(num_layers=12, embed_dim=480, **model_kwargs)
    else:
        raise NotImplementedError

    return model


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class StructureAwareAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads=16,
        gnn_type="gin",
        k_hop=1,
        use_cls_token=False,
        attn_drop=0.0,
        batch_renorm=False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cls_token = use_cls_token

        if k_hop == 0:
            self.structure_extractor = None
        else:
            self.structure_extractor = StructureExtractor(
                embed_dim,
                gnn_type=gnn_type,
                k_hop=k_hop,
                batch_renorm=batch_renorm,
                **kwargs,
            )

        self.norm = nn.LayerNorm(embed_dim)

        self.attn = Attention(embed_dim, num_heads, attn_drop=attn_drop)

    def forward(
        self, x, shift, scale, edge_index, edge_attr=None, mask=None, pos_encoding=None
    ):
        if self.structure_extractor is not None:
            x_struct = torch.zeros_like(x)

            # Cls token
            if self.use_cls_token:
                x_struct[0] = x[0]
                x_tmp, pos_encoding = self.structure_extractor(
                    x[1:][mask], edge_index, edge_attr, pos_encoding
                )
                x_struct = x_struct.to(x_tmp.dtype)
                x_struct[1:][mask] = x_tmp
            else:
                x_tmp, pos_encoding = self.structure_extractor(
                    x[mask], edge_index, edge_attr, pos_encoding
                )
                x_struct = x_struct.to(x_tmp.dtype)
                x_struct[mask] = x_tmp
            del x_tmp
            x = x + x_struct
        x = modulate(self.norm(x), shift, scale)

        return self.attn(x, padding_mask=mask), pos_encoding


class StructureExtractor(nn.Module):
    def __init__(
        self, embed_dim, gnn_type="gine", k_hop=1, batch_renorm=False, **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.gnn_type = gnn_type
        self.k_hop = k_hop
        self.kwargs = kwargs

        self.gnn = nn.ModuleList(
            [self.get_base_layer(batch_renorm=batch_renorm) for _ in range(k_hop)]
        )

    def forward(self, x, edge_index, edge_attr=None, pos_encoding=None):
        h = self.gnn[0](x, edge_index, edge_attr)
        for i in range(1, self.k_hop):
            h = self.gnn[i](h, edge_index, edge_attr) + h
        return h, pos_encoding

    def get_base_layer(self, batch_renorm=False):
        embed_dim = self.embed_dim
        if self.gnn_type == "gine":
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
                nn.BatchNorm1d(2 * embed_dim)
                if not batch_renorm
                else BatchRenorm1d(2 * embed_dim),
                nn.ReLU(True),
                nn.Linear(2 * embed_dim, embed_dim),
            )
            return gnn.GINEConv(mlp, train_eps=True, edge_dim=embed_dim)
        elif self.gnn_type == "gin":
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
                nn.BatchNorm1d(2 * embed_dim)
                if not batch_renorm
                else BatchRenorm1d(2 * embed_dim),
                nn.ReLU(True),
                nn.Linear(2 * embed_dim, embed_dim),
            )
            return gnn.GINConv(mlp, train_eps=True)
        elif self.gnn_type == "gen":
            return gnn.GENConv(embed_dim, embed_dim)
        elif self.gnn_type == "gat":
            return gnn.GATv2Conv(embed_dim, embed_dim)
        else:
            raise ValueError("Unknown gnn layer!")


def modulate(x, shift, scale):
    if x.ndim == 3:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=QuickGELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, padding_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if padding_mask is not None:
            # padding_mask = padding_mask.view(B, 1, N, 1) * padding_mask.view(B, 1, 1, N)
            padding_mask = padding_mask.view(B, 1, 1, N)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            padding_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        gnn_type="gine",
        k_hop=2,
        use_cls_token=False,
        attn_drop=0.0,
        batch_renorm=False,
        **kwargs,
    ):
        super().__init__()
        # self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = StructureAwareAttention(
            embed_dim,
            num_heads,
            gnn_type,
            k_hop,
            use_cls_token,
            attn_drop=attn_drop,
            batch_renorm=batch_renorm,
            **kwargs,
        )
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        approx_gelu = lambda: QuickGELU()
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )

    def forward(self, x, c, edge_index, edge_attr=None, mask=None, pos_encoding=None):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        attn, pos_encoding = self.attn(
            x, shift_msa, scale_msa, edge_index, edge_attr, mask, pos_encoding
        )
        x = x + (1 + gate_msa.unsqueeze(1)) * attn
        # x = x + attn
        del attn
        x = x + (1 + gate_mlp.unsqueeze(1)) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x, pos_encoding
