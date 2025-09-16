# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATConv
from einops.layers.torch import Rearrange
from torch_geometric.utils import dense_to_sparse

def weights_init_normal(m):
    if isinstance(m, GATConv):
        return
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif 'Linear' in classname:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return x + self.fn(x, **kwargs)

class PatchEmbedding(nn.Module):
    def __init__(self, num_channels=63, emb_size=40):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
    def forward(self, x):
        x = self.shallownet(x)
        return self.projection(x)

class FlattenHead(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250, num_channels=63, top_k=8, heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_channels = num_channels
        self.top_k = top_k
        self.heads = heads
        self.conv1 = GATConv(in_channels=in_channels,
                             out_channels=out_channels,
                             heads=self.heads)
        self.proj = nn.Linear(out_channels * heads, out_channels)

    def forward(self, x):
        # x: (B, 1, C, F)
        B, _, C, Fdim = x.size()
        x_for_adj = x.squeeze(1)  # (B, C, F)
        x_centered = x_for_adj - x_for_adj.mean(dim=-1, keepdim=True)
        x_norm = x_centered / (x_centered.std(dim=-1, keepdim=True) + 1e-8)
        adj_batch = torch.bmm(x_norm, x_norm.transpose(1, 2)) / Fdim  # (B, C, C)
        adj = adj_batch.mean(dim=0)  # (C, C)

        if self.top_k is not None and self.top_k < self.num_channels:
            topk_vals, _ = torch.topk(adj, self.top_k, dim=-1)
            thresh = topk_vals[:, -1].unsqueeze(-1)
            adj[adj < thresh] = 0

        edge_index, edge_weight = dense_to_sparse(adj)
        x = x.view(B * C, Fdim)
        x = self.conv1(x, edge_index, edge_attr=edge_weight)
        x = self.proj(x)
        x = x.view(B, C, -1).unsqueeze(1)  # (B, 1, C, F')
        return x

class Enc_eeg(nn.Sequential):
    def __init__(self, num_channels=63, emb_size=40):
        super().__init__(
            ResidualAdd(nn.Sequential(
                EEG_GAT(num_channels=num_channels),
                nn.Dropout(0.3),
            )),
            PatchEmbedding(num_channels, emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
