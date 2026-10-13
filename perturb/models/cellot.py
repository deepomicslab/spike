from pathlib import Path
import torch
from collections import namedtuple
from perturb.networks.icnns import ICNN

from absl import flags

FLAGS = flags.FLAGS

FGPair = namedtuple("FGPair", "f g")


def load_networks(config, **kwargs):
    def unpack_kernel_init_fxn(name="uniform", symmetric=False, **kwargs):
        if name == "normal":
            def init(*args):
                return torch.nn.init.normal_(*args, **kwargs)
        elif name == "uniform":
            init_kwargs = dict(kwargs)
            # Treat a lone `b` as a symmetric range for unconstrained A layers.
            # PyTorch's uniform_ defaults to [0, b], which makes all A weights
            # positive and blows up CellOT transport on nonnegative expression data.
            if symmetric and "a" not in init_kwargs and "b" in init_kwargs:
                init_kwargs["a"] = -init_kwargs["b"]

            def init(*args):
                return torch.nn.init.uniform_(*args, **init_kwargs)
        else:
            raise ValueError
        return init

    def initialize_icnn(net, init_config):
        a_init = unpack_kernel_init_fxn(symmetric=True, **init_config)
        w_init = unpack_kernel_init_fxn(**init_config)

        for layer in net.A:
            a_init(layer.weight)
            torch.nn.init.zeros_(layer.bias)

        for layer in net.W:
            w_init(layer.weight)

    kwargs.setdefault("hidden_units", [64] * 4)
    kwargs.update(dict(config.model))

    # eg parameters specific to g are stored in config.model.g
    kwargs.pop("name")
    if "latent_dim" in kwargs:
        kwargs.pop("latent_dim")
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})

    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    finit = dict(fkwargs.pop("kernel_init_fxn", {"name": "uniform"}))

    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gint = dict(gkwargs.pop("kernel_init_fxn", {"name": "uniform"}))

    f = ICNN(**fkwargs)
    g = ICNN(**gkwargs)
    initialize_icnn(f, finit)
    initialize_icnn(g, gint)

    if "verbose" in FLAGS and FLAGS.verbose:
        print(g)
        print(kwargs)

    return f, g


def load_opts(config, f, g):
    kwargs = dict(config.optim)
    assert kwargs.pop("optimizer", "Adam") == "Adam"
    fupd = kwargs.pop("f", {})
    gupd = kwargs.pop("g", {})
    fkwargs = kwargs.copy()
    fkwargs.update(fupd)
    fkwargs["betas"] = (fkwargs.pop("beta1", 0.9), fkwargs.pop("beta2", 0.999))
    gkwargs = kwargs.copy()
    gkwargs.update(gupd)
    gkwargs["betas"] = (gkwargs.pop("beta1", 0.9), gkwargs.pop("beta2", 0.999))

    opts = FGPair(
        f=torch.optim.Adam(f.parameters(), **fkwargs),
        g=torch.optim.Adam(g.parameters(), **gkwargs))

    return opts


def load_cellot_model(config, restore=None, **kwargs):
    f, g = load_networks(config, **kwargs)
    opts = load_opts(config, f, g)

    if restore is not None and Path(restore).exists():
        ckpt = torch.load(restore, weights_only=False)
        f.load_state_dict(ckpt["f_state"])
        opts.f.load_state_dict(ckpt["opt_f_state"])

        g.load_state_dict(ckpt["g_state"])
        opts.g.load_state_dict(ckpt["opt_g_state"])

    return (f, g), opts


def compute_loss_g(f, g, source, transport=None):
    if transport is None:
        transport = g.transport(source)

    return f(transport) - torch.multiply(source, transport).sum(-1, keepdim=True)


def compute_g_constraint(g, form=None, beta=0):
    if form is None or form == "None":
        return 0

    if form == "clamp":
        g.clamp_w()
        return 0

    elif form == "fnorm":
        if beta == 0:
            return 0

        return beta * sum(map(lambda w: w.weight.norm(p="fro"), g.W))

    raise ValueError


def compute_loss_f(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source)

    return -f(transport) + f(target)


def compute_w2_distance(f, g, source, target, transport=None):
    if transport is None:
        transport = g.transport(source).squeeze()

    with torch.no_grad():
        Cpq = (source * source).sum(1, keepdim=True) + (target * target).sum(
            1, keepdim=True
        )
        Cpq = 0.5 * Cpq

        cost = (
                f(transport)
                - torch.multiply(source, transport).sum(-1, keepdim=True)
                - f(target)
                + Cpq
        )
        cost = cost.mean()
    return cost


def numerical_gradient(param, fxn, *args, eps=1e-4):
    with torch.no_grad():
        param += eps
    plus = float(fxn(*args))

    with torch.no_grad():
        param -= 2 * eps
    minus = float(fxn(*args))

    with torch.no_grad():
        param += eps

    return (plus - minus) / (2 * eps)
