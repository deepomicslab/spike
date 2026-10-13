"""Comparator prediction export; not the current paper-model inference pipeline."""
import numpy as np
import pandas as pd
import os
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from perturb.utils.evaluate import (load_conditions, compute_knn_enrichment)
from perturb.losses.mmd import mmd_distance
from perturb.data.cell import read_single_anndata


def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])


def compute_pairwise_corrs(df):
    corr = df.corr().rename_axis(index='lhs', columns='rhs')
    return (
        corr
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .set_index(['lhs', 'rhs'])
        .squeeze()
        .rename()
        .dropna())  # Drop NaN values from zero-variance features


def compute_evaluations(iterator):
    gammas = np.logspace(1, -3, num=50)
    for ncells, nfeatures, treated, imputed in iterator:
        mut, mui = treated.mean(0), imputed.mean(0)
        stdt, stdi = treated.std(0), imputed.std(0)
        pwct = compute_pairwise_corrs(treated)
        pwci = compute_pairwise_corrs(imputed)

        # Align pairwise correlations to handle NaN values from zero-variance features
        common_idx = pwct.index.intersection(pwci.index)
        pwct_aligned = pwct.loc[common_idx]
        pwci_aligned = pwci.loc[common_idx]

        yield ncells, nfeatures, 'l2-means', np.linalg.norm(mut - mui)
        yield ncells, nfeatures, 'l2-stds', np.linalg.norm(stdt - stdi)
        yield ncells, nfeatures, 'r2-means', pd.Series.corr(mut, mui)
        yield ncells, nfeatures, 'r2-stds', pd.Series.corr(stdt, stdi)
        yield ncells, nfeatures, 'r2-pairwise_feat_corrs', pd.Series.corr(pwct_aligned, pwci_aligned)
        yield ncells, nfeatures, 'l2-pairwise_feat_corrs', np.linalg.norm(pwct_aligned - pwci_aligned)

        if treated.shape[1] < 1000:
            mmd = compute_mmd_loss(treated, imputed, gammas=gammas)
            yield ncells, nfeatures, 'mmd', mmd

            knn, enrichment = compute_knn_enrichment(imputed, treated)
            k50 = enrichment.iloc[:, :50].values.mean()
            k100 = enrichment.iloc[:, :100].values.mean()

            yield ncells, nfeatures, 'enrichment-k50', k50
            yield ncells, nfeatures, 'enrichment-k100', k100


def update_args(args, config_path):
    if config_path.startswith("./"):
        config_path = str(Path(__file__).parent) + '/' + config_path[2:]
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            dict = yaml.load(f, Loader=yaml.FullLoader)
        for key in dict.keys():
            args.__dict__[key] = dict[key]
    return args


def main(FLAGS):
    expdir = Path(FLAGS.outdir)
    setting = FLAGS.setting
    where = FLAGS.where
    embedding = FLAGS.embedding
    prefix = FLAGS.evalprefix
    if (embedding is None) or len(embedding) == 0:
        embedding = None
    if prefix is None:
        prefix = f'evals_{setting}_{where}'
    outdir = expdir / prefix
    outdir.mkdir(exist_ok=True, parents=True)

    cache = outdir / 'imputed.h5ad'
    _, treateddf, imputed = load_conditions(expdir, where, setting, embedding=embedding)
    imputed.write(cache)
    print(f'Saved imputed data to {cache}')
    return


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Export predictions from an explicit comparator run.')
    parser.add_argument('--predictions', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--n_markers', type=str, default=None)
    parser.add_argument('--n_cells', type=str, default='100,250,500,1000,1500')
    parser.add_argument('--n_reps', type=int, default=10)
    parser.add_argument('--embedding', type=str, default=None)
    parser.add_argument('--evalprefix', type=str, default=None)
    parser.add_argument('--setting', type=str, default='iid')
    parser.add_argument('--where', type=str, default='data_space')
    parser.add_argument('--via', type=str, default='')
    parser.add_argument('--subname', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
