"""
Historical architecture ablations for the published Figure 5A–D analysis.

These variants are separate from the current conditional residual model.

Variants:
- SpikeSingleDecoder: GATv2 + Single decoder + Edge features (no separate delta network)
- SpikeStandardGAT: Standard GAT + Dual decoder + Edge features
- SpikeNoEdge: GATv2 + Dual decoder + No edge features
"""

from torch import nn
from pathlib import Path
import torch.nn.functional as F
import random
import torch
import os
from collections import namedtuple
from perturb.networks.GATv2Conv import GATv2Conv
from torch_geometric.nn import GATConv

LossOutput = namedtuple('LossOutput', ['total', 'recon', 'align'])


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_optimizer(config, params):
    kwargs = dict(config.optim)
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    optim = torch.optim.AdamW(params, **kwargs)
    return optim


def dnn(dinput, doutput, hidden_units=(16, 16), activation="ReLU", dropout=0.0, batch_norm=False, **kwargs):
    if isinstance(hidden_units, int):
        hidden_units = [hidden_units]
    hidden_units = list(hidden_units)
    layer_sizes = zip([dinput] + hidden_units[:-1], hidden_units)
    if isinstance(activation, str):
        Activation = getattr(nn, activation)
    else:
        Activation = activation
    layers = list()
    for indim, outdim in layer_sizes:
        layers.append(nn.Linear(indim, outdim, **kwargs))
        if batch_norm:
            layers.append(nn.BatchNorm1d(outdim))
        layers.append(Activation())
        if dropout is not None and dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_units[-1], doutput))
    net = nn.Sequential(*layers)
    return net


# ============================================================
# Encoder variants
# ============================================================

class EncoderGATv2(nn.Module):
    """Encoder with GATv2 and edge features (default)"""
    def __init__(self, in_channels, hidden_channels, heads=1, edge_dim=3):
        super(EncoderGATv2, self).__init__()
        print(f'EncoderGATv2: incoming channel {in_channels}, edge_dim={edge_dim}')
        self.conv = GATv2Conv(in_channels, hidden_channels, edge_dim=edge_dim, heads=heads, concat=False)
        self.conv_2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=heads, concat=False)
        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'
        self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        x, attention_scores, attention_scores_unnormalized = self.conv(
            data.x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights=True)
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores_unnormalized

        x, attention_scores, attention_scores_unnormalized = self.conv_2(
            x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights=True)
        self.attention_scores_mine = attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized
        x = self.prelu(x)
        return x


class EncoderStandardGAT(nn.Module):
    """Encoder with standard GAT (PyG GATConv)"""
    def __init__(self, in_channels, hidden_channels, heads=1, edge_dim=3):
        super(EncoderStandardGAT, self).__init__()
        print(f'EncoderStandardGAT: incoming channel {in_channels}')
        # Standard GATConv from PyG - note: edge_dim handling is different
        self.conv = GATConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=heads, concat=False)
        self.conv_2 = GATConv(hidden_channels, hidden_channels, edge_dim=edge_dim, heads=heads, concat=False)
        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'
        self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        x, (edge_index, attention_scores) = self.conv(
            data.x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights=True)
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores  # Standard GAT doesn't expose unnormalized

        x, (edge_index, attention_scores) = self.conv_2(
            x, data.edge_index, edge_attr=data.edge_attr, return_attention_weights=True)
        self.attention_scores_mine = attention_scores
        self.attention_scores_mine_unnormalized = attention_scores
        x = self.prelu(x)
        return x


class EncoderNoEdge(nn.Module):
    """Encoder with GATv2 but without edge features"""
    def __init__(self, in_channels, hidden_channels, heads=1):
        super(EncoderNoEdge, self).__init__()
        print(f'EncoderNoEdge: incoming channel {in_channels}, NO edge features')
        # edge_dim=None means no edge features
        self.conv = GATv2Conv(in_channels, hidden_channels, edge_dim=None, heads=heads, concat=False)
        self.conv_2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=None, heads=heads, concat=False)
        self.attention_scores_mine_l1 = 'attention_l1'
        self.attention_scores_mine_unnormalized_l1 = 'attention_unnormalized_l1'
        self.attention_scores_mine = 'attention'
        self.attention_scores_mine_unnormalized = 'attention_unnormalized'
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, data):
        # Note: we still pass edge_attr but it will be ignored
        x, attention_scores, attention_scores_unnormalized = self.conv(
            data.x, data.edge_index, return_attention_weights=True)
        self.attention_scores_mine_l1 = attention_scores
        self.attention_scores_mine_unnormalized_l1 = attention_scores_unnormalized

        x, attention_scores, attention_scores_unnormalized = self.conv_2(
            x, data.edge_index, return_attention_weights=True)
        self.attention_scores_mine = attention_scores
        self.attention_scores_mine_unnormalized = attention_scores_unnormalized
        x = self.prelu(x)
        return x


# ============================================================
# Ablation Models
# ============================================================

class SpikeSingleDecoder(nn.Module):
    """
    Ablation: Single Decoder
    - GATv2 + Single decoder + Edge features
    - Only one decoder for both source reconstruction and target prediction
    - No separate delta network
    """

    def __init__(self, input_dim, latent_dim, hidden_units=None, dropout=0, **kwargs):
        super(SpikeSingleDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.encoder_net = EncoderGATv2(in_channels=input_dim + 1, hidden_channels=latent_dim, heads=1, edge_dim=3)
        # Single decoder for both source and target
        self.decoder_net = dnn(dinput=latent_dim, doutput=input_dim + 1, hidden_units=hidden_units[::-1], dropout=dropout)

    def encode(self, data, **kwargs):
        return self.encoder_net(data, **kwargs)

    def decode(self, code, **kwargs):
        return self.decoder_net(code, **kwargs)

    def forward(self, source, **kwargs):
        latent = self.encode(source)
        # wUse same decoder for both - this is the ablation
        source_recon = self.decode(latent, **kwargs)
        target_recon = self.decode(latent, **kwargs)  # Same output!
        return source_recon, target_recon


class SpikeStandardGAT(nn.Module):
    """
    Ablation: Standard GAT
    - Standard GAT (PyG GATConv) + Dual decoder + Edge features
    - Tests GATv2 vs GAT contribution
    """

    def __init__(self, input_dim, latent_dim, hidden_units=None, dropout=0, **kwargs):
        super(SpikeStandardGAT, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        # Use standard GAT encoder
        self.encoder_net = EncoderStandardGAT(in_channels=input_dim + 1, hidden_channels=latent_dim, heads=1, edge_dim=3)
        # Dual decoders (same as full model)
        self.decoder_net = dnn(dinput=latent_dim, doutput=input_dim + 1, hidden_units=hidden_units[::-1], dropout=dropout)
        self.delta_net = dnn(dinput=latent_dim, doutput=input_dim + 1, hidden_units=hidden_units[::-1], dropout=dropout)
        # Initialize delta with decoder weights
        self.delta_net.load_state_dict(self.decoder_net.state_dict())

    def encode(self, data, **kwargs):
        return self.encoder_net(data, **kwargs)

    def decode(self, code, **kwargs):
        return self.decoder_net(code, **kwargs)

    def delta(self, latent):
        return self.delta_net(latent)

    def forward(self, source, **kwargs):
        latent = self.encode(source)
        source_recon = self.decode(latent, **kwargs)
        target_recon = self.delta(latent)
        return source_recon, target_recon


class SpikeNoEdge(nn.Module):
    """
    Ablation: No Edge Features
    - GATv2 + Dual decoder + No edge features
    - Tests edge feature contribution
    """

    def __init__(self, input_dim, latent_dim, hidden_units=None, dropout=0, **kwargs):
        super(SpikeNoEdge, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        # Encoder without edge features
        self.encoder_net = EncoderNoEdge(in_channels=input_dim + 1, hidden_channels=latent_dim, heads=1)
        # Dual decoders
        self.decoder_net = dnn(dinput=latent_dim, doutput=input_dim + 1, hidden_units=hidden_units[::-1], dropout=dropout)
        self.delta_net = dnn(dinput=latent_dim, doutput=input_dim + 1, hidden_units=hidden_units[::-1], dropout=dropout)
        self.delta_net.load_state_dict(self.decoder_net.state_dict())

    def encode(self, data, **kwargs):
        return self.encoder_net(data, **kwargs)

    def decode(self, code, **kwargs):
        return self.decoder_net(code, **kwargs)

    def delta(self, latent):
        return self.delta_net(latent)

    def forward(self, source, **kwargs):
        latent = self.encode(source)
        source_recon = self.decode(latent, **kwargs)
        target_recon = self.delta(latent)
        return source_recon, target_recon


# ============================================================
# Loss function (shared across all variants)
# ============================================================

def compute_ablation_loss(source_recon, source, target_recon, target, alpha=0.5, scale_factor=1e-5):
    """Historical architecture-ablation loss; separate from the current residual objective."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_mask, target_x = target.mask, target.graph.x.to(device)
    source_mask, source_x = source.mask, source.graph.x.to(device)
    target_x, source_x = target_x[:, :-1], source_x[:, :-1]
    target_recon, source_recon = target_recon[:, :-1], source_recon[:, :-1]

    recon_loss = F.mse_loss(source_recon[source_mask], source_x[source_mask], reduction='mean')
    align_loss = compute_unalign_loss(target_recon[source_mask], target_x[target_mask])
    align_loss = align_loss * scale_factor

    total_loss = alpha * recon_loss + (1 - alpha) * align_loss
    return LossOutput(total_loss, alpha * recon_loss, (1 - alpha) * align_loss)


def compute_unalign_loss(imputed_tensor, target_tensor, all_ncells=[100, 250, 500], n_reps=10):
    def iterate_feature_slices(all_ncells, treated_tensor, imputed_tensor, n_reps=10):
        for ncells in all_ncells:
            if ncells > min(treated_tensor.size(0), imputed_tensor.size(0)):
                break
            for r in range(n_reps):
                idx_treated = torch.randperm(treated_tensor.size(0))[:ncells]
                idx_imputed = torch.randperm(imputed_tensor.size(0))[:ncells]
                trt = treated_tensor[idx_treated]
                imp = imputed_tensor[idx_imputed]
                yield ncells, 'all', trt, imp

    mse_vals = []
    for ncells, nfeatures, treated, imputed in iterate_feature_slices(all_ncells, target_tensor, imputed_tensor, n_reps):
        mse_val = F.mse_loss(treated, imputed, reduction='sum')
        mse_vals.append(mse_val)
    mse = torch.stack(mse_vals).mean()
    return mse


# ============================================================
# Model loading functions
# ============================================================

def load_networks(config, **kwargs):
    kwargs = kwargs.copy()
    kwargs.update(dict(config.model))
    name = kwargs.pop("name")

    model_map = {  "ablation_single_decoder": SpikeSingleDecoder,
        "ablation_standard_gat": SpikeStandardGAT,
        "ablation_no_edge": SpikeNoEdge, }

    if name not in model_map:
        raise ValueError(f"Unknown ablation model: {name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model_map[name](**kwargs).to(device)


def load_ablation_model(config, restore=None, **kwargs):
    model = load_networks(config, **kwargs)
    optim = load_optimizer(config, model.parameters())
    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optim_state"])
    return model, optim
