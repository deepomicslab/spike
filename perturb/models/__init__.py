"""Comparator and historical ablation loaders; current Spike is in conditional_residual."""

def load_cellot_model(*args, **kwargs):
    from .cellot import load_cellot_model as _load_cellot_model
    return _load_cellot_model(*args, **kwargs)


def load_autoencoder_model(*args, **kwargs):
    from .ae import load_autoencoder_model as _load_autoencoder_model
    return _load_autoencoder_model(*args, **kwargs)


def load_scgpt_model(*args, **kwargs):
    from .scgpt import load_scgpt_model as _load_scgpt_model
    return _load_scgpt_model(*args, **kwargs)


def load_prnet_model(*args, **kwargs):
    from .PRnet import load_prnet_model as _load_prnet_model
    return _load_prnet_model(*args, **kwargs)


def load_cellnest_model(*args, **kwargs):
    from .CellNest import load_cellnest_model as _load_cellnest_model
    return _load_cellnest_model(*args, **kwargs)


def load_state_model(*args, **kwargs):
    from .state import load_state_model as _load_state_model
    return _load_state_model(*args, **kwargs)


def load_stack_model(*args, **kwargs):
    from .stack import load_stack_model as _load_stack_model
    return _load_stack_model(*args, **kwargs)


def load_ablation_model(*args, **kwargs):
    from .ablation import load_ablation_model as _load_ablation_model
    return _load_ablation_model(*args, **kwargs)
