import torch
import numpy as np
import random
from torch import nn
from collections import namedtuple
from pathlib import Path
from torch import Tensor
from typing import Callable, Tuple
from torch.nn import Module, Parameter
from torch_geometric.nn.inits import reset, uniform
from perturb.networks.GATv2Conv import GATv2Conv
import os


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_optimizer(config, params):
    kwargs = dict(config.optim)
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    optim = torch.optim.AdamW(params, **kwargs)
    return optim


def load_networks(config, **kwargs):
    kwargs = kwargs.copy()
    kwargs.update(dict(config.model))
    name = kwargs.pop("name")
    if name == "CellNest":
        model = CellNEST
    else:
        raise ValueError
    kwargs.pop("device", "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model(**kwargs).to(device)


def load_cellnest_model(config, restore=None, **kwargs):
    set_seed()  # Set seed before model initialization
    model = load_networks(config, **kwargs)
    optim = load_optimizer(config, model.parameters())
    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
    return model, optim


class DeepGraphInfomax(nn.Module):
    def __init__(self, hidden_channels: int, encoder: Module, summary: Callable):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.encoder_net = encoder
        self.summary = summary
        self.corruption = corruption
        self.weight = Parameter(torch.empty(hidden_channels, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder_net)
        uniform(self.hidden_channels, self.weight)

    def forward(self, data) -> Tuple[Tensor, Tensor, Tensor]:
        pos_z = self.encoder_net(data)
        cor_data = self.corruption(data)
        neg_z = self.encoder_net(cor_data)
        summary = self.summary(pos_z, data)
        return pos_z, neg_z, summary

    def discriminate(self, z: Tensor, summary: Tensor, sigmoid: bool = True) -> Tensor:
        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(z, torch.matmul(self.weight, summary))
        return torch.sigmoid(value) if sigmoid else value

    def loss(self, pos_z: Tensor, neg_z: Tensor, summary: Tensor) -> Tensor:
        EPS = 1e-15
        pos_loss = -torch.log(self.discriminate(pos_z, summary, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.discriminate(neg_z, summary, sigmoid=True) + EPS).mean()
        return pos_loss + neg_loss

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.hidden_channels})'


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads):
        super(Encoder, self).__init__()
        print('incoming channel %d' % in_channels)
        heads = heads
        self.conv = GATv2Conv(in_channels, hidden_channels, edge_dim=3, heads=heads, concat=False)
        self.conv_2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=3, heads=heads, concat=False)
        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'
        self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        x, attention_scores, attention_scores_unnormalized = self.conv(data.x, data.edge_index,
                                                                       edge_attr=data.edge_attr,
                                                                       return_attention_weights=True)
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores_unnormalized
        x, attention_scores, attention_scores_unnormalized = self.conv_2(x, data.edge_index, edge_attr=data.edge_attr,
                                                                         return_attention_weights=True)
        self.attention_scores_mine = attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized
        x = self.prelu(x)
        return x


class CellNEST(nn.Module):
    LossComps = namedtuple("CellNESTLoss", "dgi")
    Outputs = namedtuple("CellNESTOutputs", "pos_z neg_z summary")

    def __init__(self, input_dim: int, latent_dim: int, heads: int = 1, dropout: float = 0.0):
        super().__init__()
        encoder = Encoder(input_dim, latent_dim, heads)
        self.dgi = DeepGraphInfomax(
            hidden_channels=latent_dim,
            encoder=encoder,
            summary=lambda z, data: torch.sigmoid(z.mean(dim=0)))
        self.hidden_channels = latent_dim
        self.heads = heads
        self.dropout = dropout

    def forward(self, data):
        return self.Outputs(*self.dgi(data))

    def loss(self, outputs):
        return self.dgi.loss(outputs.pos_z, outputs.neg_z, outputs.summary)


class my_data:
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr


def corruption(data):
    set_seed()  # Ensure seed is set for reproducibility in corruption
    x = data.x[torch.randperm(data.x.size(0))]
    return my_data(x, data.edge_index, data.edge_attr)
