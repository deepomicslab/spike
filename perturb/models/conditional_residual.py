"""Own-cell residual prediction with an optional bounded graph attention path."""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


def canonicalize_torch(x):
    counts = torch.expm1(x.clamp(min=0, max=20))
    return torch.log1p(counts * (10000 / counts.sum(1, keepdim=True).clamp_min(1e-12)))


class ConditionalResidual(nn.Module):
    def __init__(self, n_genes, shift, training_edges, n_lr, use_graph, width=32):
        super().__init__()
        self.use_graph = bool(use_graph)
        self.register_buffer("shift", shift.detach().clone())
        self.register_buffer("edge_mean", training_edges[:, :2].mean(0))
        self.register_buffer("edge_std", training_edges[:, :2].std(0, unbiased=False).clamp_min(1e-3))
        seen = torch.zeros(n_lr + 1, dtype=torch.bool, device=training_edges.device)
        ids = self.lr_ids(training_edges, n_lr)
        seen[ids] = True
        self.register_buffer("seen_lr", seen)
        self.n_lr = n_lr
        self.own = nn.Sequential(nn.Linear(n_genes, width), nn.LayerNorm(width), nn.GELU())
        self.lr_embedding = nn.Embedding(n_lr + 1, 4, padding_idx=n_lr)
        nn.init.normal_(self.lr_embedding.weight, std=.05)
        with torch.no_grad():
            self.lr_embedding.weight[n_lr].zero_()
        self.graph = GATv2Conv(width, width, heads=1, edge_dim=6, dropout=0., add_self_loops=True)
        self.graph_post = nn.Sequential(nn.LayerNorm(width), nn.GELU())
        self.head = nn.Linear(width, n_genes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def lr_ids(attrs, n_lr):
        values = attrs[:, 2]
        if not torch.isfinite(attrs).all() or not torch.equal(values, values.round()):
            raise ValueError("Finite edge attributes and integer LR categories required")
        if ((values < 0) | (values >= n_lr)).any():
            raise ValueError("LR category outside the fixed prior vocabulary")
        return values.long()

    def edge_features(self, attrs):
        ids = self.lr_ids(attrs, self.n_lr)
        ids = torch.where(self.seen_lr[ids], ids, self.n_lr)
        continuous = ((attrs[:, :2] - self.edge_mean) / self.edge_std).clamp(-5, 5)
        return torch.cat([continuous, self.lr_embedding(ids)], dim=1)

    def forward(self, x, edge_index, edge_attr):
        own = self.own(x)
        h = own
        if self.use_graph:
            h = own + .5 * self.graph_post(self.graph(own, edge_index, self.edge_features(edge_attr)))
        delta = self.head(h)
        return canonicalize_torch(x + self.shift + delta), delta
