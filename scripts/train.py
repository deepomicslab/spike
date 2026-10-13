"""Comparator training CLI; the current paper model is supplied as source only."""
import yaml
import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import perturb.train as perturb
from perturb.train.experiment import name_expdir

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
print(str(PROJECT_ROOT))


def normalize_repo_path(value):
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return str(path)

    if value.startswith("./"):
        return str((PROJECT_ROOT / value[2:]).resolve())
    if value.startswith("../"):
        return str((PROJECT_ROOT / value[3:]).resolve())
    return str((PROJECT_ROOT / value).resolve())


def normalize_outroot(config):
    outroot = getattr(config, 'outroot', None)
    if outroot is None:
        config.outroot = str(PROJECT_ROOT / 'results')
        return config

    outroot_path = Path(outroot)
    if outroot_path.is_absolute():
        config.outroot = str(outroot_path)
        return config

    # Project configs use relative `../results`, but training/evaluation expect
    # artifacts under the repository-local results directory.
    if outroot_path.name == 'results':
        config.outroot = str(PROJECT_ROOT / 'results')
    else:
        config.outroot = str((PROJECT_ROOT / outroot_path).resolve())
    return config


def normalize_pretrained(config):
    pretrained = getattr(config, 'pretrained', None)
    if pretrained:
        config.pretrained = normalize_repo_path(pretrained)
    return config


def update_args(args, key2):
    config_path = args.__dict__[key2]
    config_path = normalize_repo_path(config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            dict = yaml.load(f, Loader=yaml.FullLoader)
        if dict is None:
            raise ValueError(f"Config file {config_path} is empty or invalid")
        for key in dict.keys():
            args.__dict__[key] = dict[key]
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return args


def main():
    config = parse_arguments()
    if config.model['name'] == "cellot":
        train = perturb.train_cellot
    elif config.model['name'] == "scgen":
        train = perturb.train_auto_encoder
    elif config.model['name'] == "prnet":
        train = perturb.train_prnet
    elif config.model['name'] == "scgpt":
        train = perturb.train_scgpt
    elif config.model['name'] == "state":
        train = perturb.train_state
    elif config.model['name'] == "stack":
        train = perturb.train_stack
    elif config.model['name'] == "CellNest":
        train = perturb.train_CellNest
    elif config.model['name'] == "identity":
        return
    elif config.model['name'] == "random":
        return
    else:
        raise ValueError(f"Unsupported comparator model: {config.model['name']}. "
                         "Current Spike source is documented in docs/current_spike_model.md.")
    config = normalize_outroot(config)
    config = normalize_pretrained(config)
    outdir = name_expdir(config)
    outdir = outdir.resolve()
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / "config.yaml", "w") as f:
        # Convert argparse.Namespace to dict for YAML serialization
        config_dict = vars(config) if hasattr(config, '__dict__') else config
        yaml.dump(config_dict, f, default_flow_style=False)
    cachedir = outdir / "cache"
    cachedir.mkdir(exist_ok=True)
    status = cachedir / "status"
    status.write_text("running")
    try:
        train(outdir, config)
    except Exception:
        status.write_text("bugged")
        print("Training bugged")
        raise
    else:

        status.write_text("done")
        print("Training finished")
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a comparator model from an explicit configuration.')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    args = update_args(args, "config")
    return args


if __name__ == '__main__':
    main()
