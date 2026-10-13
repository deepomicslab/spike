# from collections import MutableMapping
from collections.abc import MutableMapping
import yaml
from ml_collections import ConfigDict
from pathlib import Path
import os
import re
import json
from time import localtime, strftime
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


def load_config(path, unparsed=None):
    path = Path(path)
    if path.exists():
        with open(path, "r") as f:
            content = f.read()
        try:
            # Try SafeLoader first (secure)
            loaded = yaml.load(content, yaml.SafeLoader)
        except yaml.constructor.ConstructorError:
            # Fall back to UnsafeLoader for legacy files with Python objects
            # This handles configs saved with argparse.Namespace before the fix
            import warnings
            warnings.warn(f"Loading config with UnsafeLoader (legacy format): {path}")
            loaded = yaml.load(content, yaml.UnsafeLoader)

        # Handle argparse.Namespace objects from legacy config files
        if hasattr(loaded, '__dict__') and not isinstance(loaded, dict):
            loaded = vars(loaded)

        config = ConfigDict(loaded)
    else:
        print("WARNING: config path not found")
        config = ConfigDict()
    if unparsed is not None:
        opts = parse_cli_opts(unparsed)
        config.update(opts)
    return config


def dump_config(path, config):
    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, Dumper=yaml.Dumper)
    return


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def split_rec(k, v, out, sep=".", as_dot_dict=False):
    # splitting keys in dict, calling recursively to break items on '.'
    k, *rest = k.split(sep, 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, DotDict() if as_dot_dict else dict()))
    else:
        out[k] = v


def nest_dict(d, sep=".", as_dot_dict=False):
    result = DotDict() if as_dot_dict else dict()
    for k, v in d.items():
        # for each key split_rec splits keys to form recursively nested dict
        split_rec(k, v, result, sep=sep, as_dot_dict=as_dot_dict)
    return result


def flat_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flat_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def symlink_to_logfile(loglink):
    loglink = Path(loglink)

    lsb_outputfile = os.getenv("LSB_OUTPUTFILE")
    jobid = os.getenv("LSB_JOBID")
    if lsb_outputfile is None or jobid is None:
        return

    logpath = Path(lsb_outputfile).resolve()
    if lsb_outputfile.endswith("/") or logpath.is_dir():
        logpath = logpath / f"{jobid}.out"

    if loglink.exists():
        loglink.unlink()
    try:
        loglink.symlink_to(logpath)
    except FileExistsError:
        pass

    with open(loglink.with_name(".loghistory"), "a") as fp:
        fp.write(f"{logpath}\n")

    return


def parse_cli_opts(args):
    def parse(argiter):
        for opt in argiter:
            if "=" not in opt:
                value = next(argiter)
                key = re.match(r"^--config\.(?P<key>.*)", opt)["key"]

            else:
                match = re.match(r"^--config\.(?P<key>.*)=(?P<value>.*)", opt)
                key, value = match["key"], match["value"]

            value = yaml.load(value, Loader=yaml.UnsafeLoader)
            yield key, value

    opts = dict()
    if len(args) == 0:
        return opts

    argiter = iter(args)
    for key, val in parse(argiter):
        *tree, leaf = key.split(".")
        lut = opts
        for k in tree:
            lut = lut.setdefault(k, dict())
        lut[leaf] = val

    return opts


def config_from_wandb(path):
    config = yaml.load(open(path), yaml.UnsafeLoader)
    del config["wandb_version"]
    del config["_wandb"]
    for key, val in config.items():
        val = val["value"]
        config[key].pop("desc", None)
        config[key].pop("value", None)
        config[key] = val

    return ConfigDict(nest_dict(config))


def parse_config_cli(path, args):
    if isinstance(path, list):
        config = ConfigDict()
        for arg in path:
            config.update(yaml.load(open(arg), Loader=yaml.RoundTripLoader))
    else:
        config = load_config(path)

    opts = parse_cli_opts(args)
    config.update(opts)

    return config


def compile_extraction_from_path(*keys):
    regex = {
        key: re.compile(rf'{key}-(\w+)/')
        for key
        in keys
    }

    def extract(path):
        path = str(path)

        def iterate():
            for k, r in regex.items():
                try:
                    v, *_ = r.findall(path)
                except ValueError:
                    v = None

                yield k, v

        return dict(iterate())

    return extract


def read_list(arg):
    """Read list from file or return as-is if already a list."""
    if isinstance(arg, str):
        arg = Path(arg)
        assert arg.exists()
        lst = arg.read_text().split()
    else:
        lst = arg
    return list(lst)


def generate_pseudotime(adata, method='pca', n_components=10, n_neighbors=15, random_state=42):
    """
    Generate pseudotime for cells when not provided.

    Args:
        adata: AnnData object
        method: Method to generate pseudotime. Options:
            - 'pca': Use first PC (fastest, simplest)
            - 'diffusion': Use diffusion pseudotime (requires scanpy)
            - 'uniform': Use uniform distribution (baseline)
        n_components: Number of components for PCA/diffusion
        n_neighbors: Number of neighbors for diffusion map
        random_state: Random seed

    Returns:
        numpy array of pseudotime values (normalized to [0, 1])
    """
    import scanpy as sc
    from sklearn.decomposition import PCA

    if method == 'uniform':
        # Simple uniform distribution as baseline
        n_cells = adata.n_obs
        pseudotime = np.linspace(0, 1, n_cells)
        np.random.seed(random_state)
        np.random.shuffle(pseudotime)
        return pseudotime.astype(np.float32)

    elif method == 'pca':
        # Use first principal component as pseudotime
        # This captures the main source of variation in the data
        X = adata.X if not sparse.issparse(adata.X) else adata.X.toarray()
        pca = PCA(n_components=min(n_components, min(X.shape)), random_state=random_state)
        pca.fit(X)
        # Use first PC, normalize to [0, 1]
        pseudotime = pca.transform(X)[:, 0]
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-8)
        return pseudotime.astype(np.float32)

    elif method == 'diffusion':
        # Use diffusion pseudotime via scanpy
        # This is more biologically meaningful but slower
        adata_tmp = adata.copy()
        sc.pp.pca(adata_tmp, n_comps=n_components, random_state=random_state)
        sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, random_state=random_state)
        sc.tl.diffmap(adata_tmp, n_comps=n_components)
        # Use first diffusion component
        pseudotime = adata_tmp.obsm['X_diffmap'][:, 0]
        pseudotime = (pseudotime - pseudotime.min()) / (pseudotime.max() - pseudotime.min() + 1e-8)
        return pseudotime.astype(np.float32)

    else:
        raise ValueError(f"Unknown pseudotime generation method: {method}")


def split_cell_data_train_test(data, groupby=None, random_state=0, holdout=None, subset=None, **kwargs):
    """Split cell data into train/test sets."""
    split = pd.Series(None, index=data.obs.index, dtype=object)
    groups = {None: data.obs.index}
    if groupby is not None:
        groups = data.obs.groupby(groupby, observed=False).groups
    for key, index in groups.items():
        # Arrow-backed pandas indexes can break sklearn's fancy indexing;
        # convert to a plain NumPy array before splitting.
        index = np.asarray(index)
        trainobs, testobs = train_test_split(index, random_state=random_state, **kwargs)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"
    return split


def split_cell_data_train_test_eval(
        data, test_size=0.15, eval_size=0.15,
        groupby=None, random_state=0, holdout=None, **kwargs):
    """Split cell data into train/test/eval sets."""
    split = pd.Series(None, index=data.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    groups = {None: data.obs.loc[split != "ood"].index}
    if groupby is not None:
        groups = data.obs.loc[split != "ood"].groupby(groupby).groups

    for key, index in groups.items():
        training, evalobs = train_test_split(
            index, random_state=random_state, test_size=eval_size)

        trainobs, testobs = train_test_split(training, random_state=random_state, test_size=test_size)

        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
        split.loc[evalobs] = "eval"

    return split


def split_cell_data_toggle_ood(data, holdout, key, mode, random_state=0, **kwargs):
    """Hold out ood sample, coordinated with iid split

    ood sample defined with key, value pair

    for ood mode: hold out all cells from a sample
    for iid mode: include half of cells in split
    """
    split = split_cell_data_train_test(data, random_state=random_state, **kwargs)

    if not isinstance(holdout, list):
        value = [holdout]

    ood = data.obs_names[data.obs[key].isin(value)]
    trainobs, testobs = train_test_split(ood, random_state=random_state, test_size=0.5)

    if mode == "ood":
        split.loc[trainobs] = "ignore"
        split.loc[testobs] = "ood"

    elif mode == "iid":
        split.loc[trainobs] = "train"
        split.loc[testobs] = "ood"

    else:
        raise ValueError

    return split


def split_cell_data(data, name="train_test", **kwargs):
    """Split cell data based on specified method."""
    if name == "train_test":
        split = split_cell_data_train_test(data, **kwargs)
    elif name == "toggle_ood":
        split = split_cell_data_toggle_ood(data, **kwargs)
    elif name == "train_test_eval":
        split = split_cell_data_train_test_eval(data, **kwargs)
    elif name == 'splited':
        split = data.obs[kwargs['obs_key']].map({'valid': 'test', 'train': 'train'})
    else:
        raise ValueError
    return split.astype("category")
