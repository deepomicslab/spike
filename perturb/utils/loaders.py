import perturb.models
from perturb.data.cell import load_cell_data


def load_data(config, **kwargs):
    data_type = config.data['type']
    if data_type in ["cell", "cell-merged", "tupro-cohort"]:
        loadfxn = load_cell_data
    else:
        raise ValueError
    return loadfxn(config, **kwargs)


def load_model(config, restore=None, **kwargs):
    name = config.model['name']
    if name == "cellot":
        loadfxn = perturb.models.load_cellot_model
    elif name == "scgen":
        loadfxn = perturb.models.load_autoencoder_model
    elif name == "scgpt":
        loadfxn = perturb.models.load_scgpt_model
    elif name == "state":
        loadfxn = perturb.models.load_state_model
    elif name == "stack":
        loadfxn = perturb.models.load_stack_model
    elif name == "prnet":
        loadfxn = perturb.models.load_prnet_model
    elif name == "CellNest":
        loadfxn = perturb.models.load_cellnest_model
    elif name == "cae":
        loadfxn = perturb.models.load_autoencoder_model
    # Ablation models
    elif name in ["ablation_single_decoder", "ablation_standard_gat", "ablation_no_edge"]:
        loadfxn = perturb.models.load_ablation_model
    else:
        raise ValueError(f"Unknown model: {name}")
    return loadfxn(config, restore=restore, **kwargs)


def load(config, restore=None, include_model_kwargs=False, **kwargs):
    loader, model_kwargs = load_data(config, include_model_kwargs=True, **kwargs)
    model, opt = load_model(config, restore=restore, **model_kwargs)
    if include_model_kwargs:
        return model, opt, loader, model_kwargs
    return model, opt, loader
