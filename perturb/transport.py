import anndata
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch.autograd import Variable
from torch.distributions import normal
import torch.nn.functional as F


def transport(config, model, dataset, return_as="anndata", dosage=None, **kwargs):
    name = config.model.get("name", "cellot")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Graph-based models need GraphDataLoader
    graph_models = ['ablation_single_decoder', 'ablation_standard_gat', 'ablation_no_edge']
    if name in graph_models:
        loader = GraphDataLoader(dataset, batch_size=1, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    # Historical architecture ablations return reconstruction and target tensors.
    ablation_models = ['ablation_single_decoder', 'ablation_standard_gat', 'ablation_no_edge']

    if name not in ablation_models:
        inputs = next(iter(loader))
    if name == "cellot":
        outputs = transport_cellot(model, inputs)
    elif name == "scgen":
        outputs, shift = transport_scgen(model, inputs.to(device),
                                         source=config.data.source,
                                         target=config.data.target,
                                         **kwargs)
        outputs = outputs.cpu()
        # Ensure output is 2D (squeeze extra dimensions if needed)
        if outputs.dim() == 3:
            outputs = outputs.squeeze()
    elif name == "scgpt":
        feature, input = inputs
        outputs, shift = transport_scgpt(model, feature.to(device), source=config.data.source,
                                         target=config.data.target, **kwargs)
        outputs = outputs.cpu()
    elif name in ['ablation_single_decoder', 'ablation_standard_gat', 'ablation_no_edge']:
        _, outputs = transport_ablation(model, dataset.graph.to(device))
        outputs = outputs[:, :-1]
        outputs = outputs.cpu()
    elif name == "state":
        feature, input = inputs
        outputs, shift = transport_state(model, feature.to(device), source=config.data.source,
                                         target=config.data.target, **kwargs)
        outputs = outputs.cpu()
    elif name == "cae":
        outputs = transport_cae(model, inputs, target=config.data.target)
    elif name == "stack":
        feature, input = inputs
        outputs, shift = transport_stack(model, feature.to(device), source=config.data.source,
                                         target=config.data.target, **kwargs)
        outputs = outputs.cpu()
    else:
        raise ValueError
    if dosage is not None:
        outputs = (1 - dosage) * inputs + dosage * outputs
    if return_as == "anndata":
        try:
            item = outputs.detach().numpy()
        except AttributeError:
            item = outputs
        # Ensure item is 2D for AnnData
        if item.ndim == 3:
            item = item.squeeze()
        outputs = anndata.AnnData(
            item,
            obs=dataset.adata.obs.copy(),
            var=dataset.adata.var.copy())
    return outputs


def transport_cellot(model, inputs):
    f, g = model
    g.eval()
    outputs = g.transport(inputs.requires_grad_(True))
    return outputs


def transport_scgen(model, inputs, source, target, decode=True):
    model.eval()
    if target not in model.code_means:
        target = 'control'
    shift = model.code_means[target] - model.code_means[source]
    codes = model.encode(inputs)
    if not decode:
        return codes + shift
    outputs = model.decode(codes + shift)
    return outputs, shift


def transport_scgpt(model, feature, source, target, decode=True):
    model.eval()
    shift = model.code_means[target] - model.code_means[source]
    codes = model.encode(feature)
    if not decode:
        return codes + shift
    outputs = model.decode(codes + shift)
    return outputs, shift


def transport_state(model, feature, source, target, decode=True):
    model.eval()
    shift = model.code_means[target] - model.code_means[source]
    codes = model.encode(feature)
    if not decode:
        return codes + shift
    outputs = model.decode(codes + shift)
    return outputs, shift


def transport_stack(model, feature, source, target, decode=True):
    model.eval()
    if target not in model.code_means:
        target = 'control'
    shift = model.code_means[target] - model.code_means[source]
    codes = model.encode(feature)
    if not decode:
        return codes + shift
    outputs = model.decode(codes + shift)
    return outputs, shift


def transport_nest(model, inputs):
    model.eval()
    outputs, _ = model(inputs)
    return outputs


def transport_ablation(model, inputs):
    model.eval()
    outputs, target = model(inputs)
    return outputs, target


def transport_cae(model, inputs, target):
    model.eval()
    target_code = model.conditions.index(target)
    outputs = model.outputs(inputs, decode_as=target_code).recon
    return outputs


def transport_prnet(model, inputs, decode=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_noise(batch_size, shape, volatile=False):
        tensor = torch.randn(batch_size, shape)
        noise = Variable(tensor, volatile)
        noise = noise.to(device, dtype=torch.float32)
        return noise

    model.eval()
    (control, target), encode_label = inputs['features'], inputs['label']
    control = control.to(device, dtype=torch.float32)
    encode_label = encode_label.to(device, dtype=torch.float32)
    b_size = control.size(0)
    noise = make_noise(b_size, 10)
    gene_reconstructions = model(control, encode_label, noise)
    dim = gene_reconstructions.size(1) // 2
    gene_means = gene_reconstructions[:, :dim]
    gene_vars = gene_reconstructions[:, dim:]
    gene_vars = F.softplus(gene_vars)
    dist = normal.Normal(
        torch.clamp(torch.Tensor(gene_means), min=1e-3, max=1e3),
        torch.clamp(torch.Tensor(gene_vars.sqrt()), min=1e-3, max=1e3))
    outputs = dist.sample()
    if not decode:
        return outputs
    return outputs
