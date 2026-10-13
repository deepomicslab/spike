from copy import deepcopy
import anndata
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import numpy as np
from perturb.losses.mmd import mmd_distance
from perturb.transport import transport
from perturb.utils import load_config
from perturb.utils.loaders import load, load_data, load_model


# ============================================
# 延迟导入模型特定的函数
# ============================================
def _lazy_import_model_func(module_name, func_name):
    """延迟导入模型函数"""
    module = __import__(f'perturb.models.{module_name}', fromlist=[func_name])
    return getattr(module, func_name)


def compute_knn_enrichment(pushfwd, treated, return_joint=False, ncells=None):
    if ncells is None:
        ncells = min(len(pushfwd), len(treated))
    assert ncells <= len(pushfwd)
    assert ncells <= len(treated)

    pushfwd = pushfwd.sample(n=ncells)
    treated = treated.sample(n=ncells)

    joint = pd.concat((pushfwd, treated), axis=0)

    labels = pd.concat(
        (pd.Series("pushfwd", index=pushfwd.index),
         pd.Series("treated", index=treated.index))).astype("category")

    n_neighbors = min([251, ncells])
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(joint)
    dists, knn = model.kneighbors(pushfwd)
    # assert np.all(knn[:, 0] == np.arange(len(knn)))
    knn = pd.DataFrame(knn[:, 1:], index=pushfwd.index)

    enrichment = knn.applymap(lambda x: labels.iloc[x] == "pushfwd")
    knn = knn.applymap(lambda x: labels.index[x])

    if return_joint:
        return knn, enrichment, joint
    return knn, enrichment


def compute_drug_signature_differences(control, treated, pushfwd):
    base = control.mean(0)
    true = treated.mean(0) - base
    pred = pushfwd.mean(0) - base
    diff = true - pred
    return diff


def compute_mmd_df(target, transport, gammas=None, subsample=False, ncells=None, nreps=5):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    gammas = list(gammas)

    def compute(ncells=None):
        for g in tqdm(gammas * nreps, desc="mmd", leave=False):
            mmd = mmd_distance(
                target if ncells is None else target.sample(ncells),
                transport if ncells is None else transport.sample(ncells), g)
            yield g, mmd

    if subsample:
        if ncells is None:
            ncells = min(len(target), len(transport))
        else:
            ncells = min(len(target), len(transport), ncells)
    elif ncells is not None:
        assert ncells <= min(len(target), len(transport))
    mmd = pd.DataFrame(compute(ncells), columns=["gamma", "mmd"])
    return mmd


def compute_r2(treated, impute, all_ncells, n_reps=10):
    def iterate_feature_slices(all_ncells, treateddf, imputeddf, n_reps=10):
        for ncells in all_ncells:
            if ncells > min(len(treateddf), len(imputeddf)):
                break
            for r in range(n_reps):
                trt = treateddf.sample(ncells)
                imp = imputeddf.sample(ncells)
                yield ncells, 'all', trt, imp

    def comput_corr(iterator):
        for ncells, nfeatures, treated, imputed in iterator:
            yield ncells, nfeatures, pd.Series.corr(treated.mean(0), imputed.mean(0))

    evals = pd.DataFrame(comput_corr(iterate_feature_slices(all_ncells, treated, impute, n_reps)),
                         columns=['ncells', 'nfeatures', 'value'])
    r2 = evals[evals['nfeatures'] == 'all']['value'].mean()
    return r2


def mean_rowwise_rmse(treated, impute, all_ncells, n_reps):
    def iterate_feature_slices(all_ncells, treateddf, imputeddf, n_reps=10):
        for ncells in all_ncells:
            if ncells > min(len(treateddf), len(imputeddf)):
                break
            for r in range(n_reps):
                trt = treateddf.sample(ncells)
                imp = imputeddf.sample(ncells)
                yield ncells, 'all', trt, imp

    def comput_mrrmse(iterator):
        for ncells, nfeatures, treated, imputed in iterator:
            treated, imputed = treated.values, imputed.values
            yield ncells, nfeatures, np.sqrt(np.square(treated - imputed).mean(axis=1)).mean()

    evals = pd.DataFrame(comput_mrrmse(iterate_feature_slices(all_ncells, treated, impute, n_reps)),
                         columns=['ncells', 'nfeatures', 'value'])
    mrrmse = evals[evals['nfeatures'] == 'all']['value'].mean()
    return mrrmse


def patch_scgen_shift(config, model):
    assert config.model.name == "scgen"
    loader = (load_data(config))
    labels = loader.train.dataset.adata.obs[config.data.condition]
    _lazy_import_model_func('ae', 'compute_scgen_shift')(model, loader.train.dataset, labels=labels)
    return


def patch_scgpt_shift(config, model):
    assert config.model.name == "scgpt"
    loader = (load_data(config))
    labels = loader.train.dataset.adata.obs[config.data.condition]
    _lazy_import_model_func('scgpt', 'compute_scgpt_shift')(model, loader.train.dataset, labels=labels)
    return


def patch_state_shift(config, model):
    assert config.model.name == "state"
    loader = (load_data(config))
    labels = loader.train.dataset.adata.obs[config.data.condition]
    _lazy_import_model_func('state', 'compute_state_shift')(model, loader.train.dataset, labels=labels)
    return


def patch_stack_shift(config, model):
    assert config.model.name == "stack"
    loader = (load_data(config))
    labels = loader.train.dataset.adata.obs[config.data.condition]
    _lazy_import_model_func('stack', 'compute_stack_shift')(model, loader.train.dataset, labels=labels)
    return


def load_projectors(aedir, embedding, where):
    if embedding is None:
        def encode(df):
            return df

        def decode(df):
            return df

        return encode, decode

    config = load_config(aedir / "config.yaml")
    ae, *_, dataset = load(
        config,
        restore=aedir / "cache" / "model.pt",
        split_on=["transport"],
        return_as="dataset")
    ae.eval()
    patch_scgen_shift(config, ae)

    features = dataset.source.adata.var_names

    try:
        pcs = dataset.source.adata.varm["PCs"]

    except KeyError:
        if "dimension_reduction" in config.data:
            tmp = deepcopy(config)
            del tmp.data.dimension_reduction
            dataset = load_data(tmp, split_on=[], return_as="anndata")
            pcs = dataset.varm["PCs"]
            features = dataset.var_names

    if embedding == "pca":

        def encode(df):
            return df @ pcs

        def decode(df):
            df = df @ pcs.T
            df.columns = features
            return df

    elif embedding == "ae":

        def encode(df):
            codes = ae.encode(torch.Tensor(df.values)).detach().numpy()
            return pd.DataFrame(codes, index=df.index)

        def decode(df):
            recon = ae.decode(torch.Tensor(df.values)).detach().numpy()
            return pd.DataFrame(recon, index=df.index, columns=features)

    else:
        raise ValueError

    return encode, decode


def read_embedding_context(config):
    embedding = None
    if "ae_emb" in config.data:
        embedding = "ae"
    elif "dimension_reduction" in config.data:
        assert config.data.dimension_reduction.name == "pca"
        embedding = "pca"

    return embedding


def load_all_inputs(config, setting, embedding, where):
    if config.model.name in ['prnet']:
        dataset, model_kwargs = load_data(config,
                                          split_on=["split"],
                                          return_as="dataset",
                                          include_model_kwargs=True)
        to_pushfwd = dataset.test
    else:
        dataset, model_kwargs = load_data(config,
                                          split_on=["split", "transport"],
                                          return_as="dataset",
                                          include_model_kwargs=True)
        if setting == "iid":
            to_pushfwd = dataset.test.source
        elif setting == "ood":
            to_pushfwd = dataset.ood.source
        else:
            raise ValueError(f"unknown setting, {setting} must be [iid, ood]")
        if config.model.name == "emergae":
            to_pushfwd = dataset.test
    # reload control and treated as per embedding & where
    if embedding == "pca" and where == "data_space":
        assert config.data.dimension_reduction.name == "pca"
        config = deepcopy(config)
        del config.data.dimension_reduction
        dataset = load_data(config, split_on=["split", "transport"], return_as="dataset")
        if config.model.name == "identity":
            to_pushfwd = dataset.test.source
    elif config.model.name == "cellot" and embedding == "ae" and where == "data_space":
        assert "ae_emb" in config.data
        config = deepcopy(config)
        del config.data.ae_emb
        dataset = load_data(
            config, split_on=["split", "transport"], return_as="dataset")
    if config.model.name == 'prnet':
        treated = dataset.test.adata.to_df()
        control = dataset.test.control.to_df()
    elif setting == "iid":
        control = dataset.test.source.adata.to_df()
        treated = dataset.test.target.adata.to_df()
    elif setting == "ood":
        control = dataset.ood.source.adata.to_df()
        if "target" in dataset.ood.keys():
            treated = dataset.ood.target.adata.to_df()
        else:
            treated = dataset.test.target.adata.to_df()
    else:
        raise ValueError(f"unknown setting, {setting} must be [iid, ood]")
    obs = load_data(config, split_on=[], return_as="anndata").obs
    return control, treated, to_pushfwd, obs, model_kwargs


def compute_mean_effect(config):
    dataset = load_data(
        config,
        return_as="dataset",
        include_model_kwargs=False,
        split_on=["split", "transport"])

    cmu = dataset["train"]["source"].adata.to_df().mean(0)
    tmu = dataset["train"]["target"].adata.to_df().mean(0)
    return tmu - cmu


def grab_treated_cells_for_random_model(config, setting):
    if setting == "ood":
        assert config.datasplit.name == "toggle_ood"

        dataset = load_data(
            config, return_as="dataset",
            include_model_kwargs=False,
            split_on=["transport", config.datasplit.key])
        cells = dataset["target"][config.datasplit.holdout].adata.to_df()

    else:
        dataset = load_data(config, return_as="dataset",
                            include_model_kwargs=False,
                            split_on=["transport"], )
        cells = dataset["target"].adata.to_df()

    cells = cells.sample(frac=1, random_state=1)
    treated, imputed = cells.iloc[: len(cells) // 2], cells.iloc[len(cells) // 2:]
    return treated, imputed


def require_trained_checkpoint(expdir):
    checkpoint = expdir / "cache" / "model.pt"
    if checkpoint.exists():
        return checkpoint

    invalid_backups = sorted(
        expdir.glob("cache.invalid.*/model.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    backup_hint = ""
    if invalid_backups:
        backup_hint = f" Latest invalid backup: {invalid_backups[0]}"

    raise FileNotFoundError(
        f"Missing trained checkpoint at {checkpoint}. "
        "Evaluating without a checkpoint would export a randomly initialized model."
        f"{backup_hint}"
    )


def load_conditions(expdir, where, setting, embedding=None):
    if embedding is None and (expdir.parent / "model-cellot" / "config.yaml").exists():
        try:
            embedding = read_embedding_context(load_config(expdir.parent / "model-cellot" / "config.yaml"))
        except Exception as e:
            print(f"Warning: Could not load cellot config for embedding context: {e}")
            embedding = None
    assert embedding in {None, "pca", "ae"}
    encode, decode = load_projectors(expdir.parent / "model-scgen", embedding, where)
    config = load_config(expdir / "config.yaml")
    if "ae_emb" in config.data:
        config.data.ae_emb.path = str(expdir.parent / "model-scgen")
    control, treated, to_pushfwd, obs, model_kwargs = load_all_inputs(config, setting, embedding, where)
    if config.model.name != "cellot" and embedding == "ae" and where == "latent_space":
        control = encode(control)
        treated = encode(treated)
    if config.model.name == "identity":
        imputed = to_pushfwd.adata.to_df()
        if embedding == "ae" and where == "latent_space":
            imputed = encode(imputed)
    elif config.model.name == "random":
        treated, imputed = grab_treated_cells_for_random_model(config, setting)
        assert pd.Index.intersection(imputed.index, treated.index).size == 0
        if len(imputed) > len(to_pushfwd):
            imputed = imputed.sample(len(to_pushfwd), random_state=0)
        if where == "latent_space":
            imputed = encode(imputed)
            treated = encode(treated)
    elif config.model.name == "average":
        effect = compute_mean_effect(config)
        imputed = to_pushfwd.adata.to_df() + effect
        if where == "latent_space":
            imputed = encode(imputed)
    else:
        assert config.model.name in {"scgen", "cae", "cellot", "scgpt", "state", "prnet",
                                      "stack", "ablation_single_decoder", "ablation_standard_gat", "ablation_no_edge"}
        checkpoint = require_trained_checkpoint(expdir)
        model, *_ = load_model(config, restore=checkpoint, **model_kwargs)
        if config.model.name == "scgen" and not hasattr(model, "code_means"):
            patch_scgen_shift(config, model)
        elif config.model.name == "scgpt" and not hasattr(model, "code_means"):
            patch_scgpt_shift(config, model)
        elif config.model.name == "state" and not hasattr(model, "code_means"):
            patch_state_shift(config, model)
        elif config.model.name == "stack" and not hasattr(model, "code_means"):
            patch_stack_shift(config, model)
        else:
            pass
        if (config.model.name == "scgen" and embedding == "ae" and where == "latent_space"):
            imputed = transport(config, model, to_pushfwd, decode=False, return_as=None)
            imputed = pd.DataFrame(imputed.detach().numpy(), index=to_pushfwd.adata.obs_names)
        else:
            imputed = transport(config, model, to_pushfwd).to_df()
        if config.model.name == "cellot" and where == "data_space":
            imputed = decode(imputed)
            # CellOT predicts in log-normalized expression space for these runs,
            # so negative values are outside the valid data range.
            imputed = imputed.clip(lower=0)
        elif (config.model.name != "cellot"
              and embedding == "pca"
              and where == "data_space"):
            imputed = decode(imputed)
        elif (config.model.name == "cae" and embedding == "ae" and where == "latent_space"):
            imputed = encode(imputed)
    imputed = anndata.AnnData(imputed, obs=obs.loc[imputed.index])
    return control, treated, imputed
