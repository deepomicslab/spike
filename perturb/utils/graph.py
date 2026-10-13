import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from collections import defaultdict
from joblib import Parallel, delayed
import time
import anndata
import random
from pathlib import Path
from tqdm import tqdm

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_sketched_cci(adata, num_neighbors=5, is_undirected=True):
    # Convert view to actual AnnData to avoid ImplicitModificationWarning
    if adata.is_view:
        adata = adata.copy()
    if 'spatial' in adata.obsm:
        coords = adata.obsm["spatial"]
    elif 'X_pca' in adata.obsm:
        coords = adata.obsm["X_pca"]
    else:
        if 'array_row' in adata.obs:
            coords = adata.obs[['array_row', 'array_col']].values
        else:
            sc.pp.pca(adata)
            coords = adata.obsm["X_pca"]
    nn = NearestNeighbors(n_neighbors=num_neighbors + 1, n_jobs=-1)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    n_cells = coords.shape[0]
    rows = np.repeat(np.arange(n_cells), num_neighbors)
    cols = indices[:, 1:].flatten()
    adj = sp.coo_matrix(
        (np.ones_like(rows), (rows, cols)),
        shape=(n_cells, n_cells))
    if is_undirected:
        adj = adj + adj.T
    adata.obsp['knn_adj'] = adj.tocsr()
    return adata, distances


def get_sketched_cci_commot(adata, num_neighbors=5, is_undirected=True):
    # Convert view to actual AnnData to avoid ImplicitModificationWarning
    if adata.is_view:
        adata = adata.copy()
    import torch
    if 'spatial' in adata.obsm:
        coords = adata.obsm["spatial"]
    elif 'X_pca' in adata.obsm:
        coords = adata.obsm["X_pca"]
    else:
        if 'array_row' in adata.obs:
            coords = adata.obs[['array_row', 'array_col']].values
        else:
            sc.pp.pca(adata)
            coords = adata.obsm["X_pca"]
    nn = NearestNeighbors(n_neighbors=num_neighbors + 1, n_jobs=-1)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)
    n_cells = coords.shape[0]
    rows = np.repeat(np.arange(n_cells), num_neighbors)
    cols = indices[:, 1:].flatten()
    adj = sp.coo_matrix(
        (np.ones_like(rows), (rows, cols)),
        shape=(n_cells, n_cells))
    if is_undirected:
        adj = adj + adj.T
    adata.obsp['knn_adj'] = adj.tocsr()

    # dist_data = distances[:, 1:].flatten()  # 提取 k-neighbors 的真实距离
    # dist_matrix_sparse = sp.coo_matrix(
    #     (dist_data, (rows, cols)),  # 用 距离 作为数据
    #     shape=(n_cells, n_cells)
    # )
    # if is_undirected:
    #     dist_matrix_sparse = dist_matrix_sparse + dist_matrix_sparse.T  # 同样使其对称
    # adata.obsp['spatial_distance'] = dist_matrix_sparse.toarray()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d_matrix = torch.tensor(distances, dtype=torch.float32, device=device)
    d_sorted, _ = d_matrix.sort()
    d_cut = torch.median(d_sorted[:, num_neighbors])
    return d_cut


def get_commot_cci(adata):
    adj = np.empty((adata.n_obs, adata.n_obs), dtype=object)
    for key in adata.obsp.keys():
        if any(['commot-' not in key, 'pathway' in key, 'total' in key]):
            continue

        lr = key.replace('commot-', '')
        row, col = np.where(adata.obsp[key].toarray())
        for i, j in zip(row, col):
            if adj[i, j] is None:
                adj[i, j] = [lr]
            else:
                adj[i, j].append(lr)
    row, col = np.where(adj)  # not None
    for i, j in zip(row, col):
        ls = adj[i, j]
        for lr in ls:
            key = f'commot-{lr}'
            assert adata.obsp[key][i, j] != 0
    adata.obsp['commot_adj'] = csr_matrix(np.where(adj, 1, 0))
    return adata


def get_commot_lrs(adata, score, save_path):
    save_path = save_path.replace('./', '../')
    exclude_keys = {'commot-toyDB-total-total', 'knn_adj', 'spatial_distance'}
    lrs = [key.replace('commot-toyDB-', '') for key in adata.obsp.keys() if key not in exclude_keys]
    if 'commot_expr' in adata.obsm.keys():
        expr_array = adata.obsm['commot_expr'].A
        print(f'expr_array shape: {expr_array.shape}')
    else:
        expr_array = np.zeros((adata.n_obs, len(lrs)))

        def process_lr(lr):
            matrix = adata.obsp[f'commot-toyDB-{lr}']
            return matrix.sum(axis=1).A1

        start = time.time()
        results = Parallel(n_jobs=6)(delayed(process_lr)(lr) for col_idx, lr in enumerate(lrs))
        print(f'Elapsed time: {time.time() - start:.2f}s')
        for idx, result in enumerate(results):
            expr_array[:, idx] = result
        adata.obsm['commot_expr'] = sp.csr_matrix(expr_array)
        adata.write(save_path)
        print(f'expr_array shape: {expr_array.shape}')
    ligand_ids = [lr.split('-')[0] for lr in lrs]
    scored_lrs = score.reindex(ligand_ids, fill_value=0)['score'].values
    adata.obsm['commot_expr'] = sp.csr_matrix(expr_array)
    expr = pd.DataFrame(expr_array * scored_lrs, columns=lrs, index=adata.obs_names)
    adata.obsm['scored_commot_expr'] = sp.csr_matrix(expr.values)
    return adata


def get_generated_lrs(adata, config):
    control_mask = adata.obs["drug"] == "control"
    source = adata[control_mask]
    drug_mask = adata.obs["drug"] == config.drug
    target = adata[drug_mask]
    source, _ = get_sketched_cci(source, config.k)
    target, _ = get_sketched_cci(target, config.k)
    if config.model['name'] == 'nest':
        source = run_nest_preprocess(source)
        target = run_nest_preprocess(target)
    else:
        source = run_nest_preprocess(source, 'high')
        target = run_nest_preprocess(target, 'high')
    X = np.concatenate([source.X, target.X], axis=0)
    obs_item = pd.concat([source.obs, target.obs])
    ad = anndata.AnnData(X=X, obs=obs_item, var=source.var, dtype=np.float32)
    for key in source.uns.keys():
        if type(source.uns[key]) is pd.DataFrame:
            ad.uns[key] = pd.concat([source.uns[key], target.uns[key]], axis=0)
        elif type(source.uns[key]) is dict:
            ad.uns[key] = {**source.uns[key], **target.uns[key]}
        elif type(source.uns[key]) is list:
            ad.uns[key] = source.uns[key] + target.uns[key]
            ad.uns[key + '_source_vs_target'] = str(len(source.uns[key])) + ', ' + str(len(target.uns[key]))
        else:
            print(f'Unsupported type for key {key}: {type(source.uns[key])}')
    ad.obs = ad.obs.reindex(adata.obs.index)
    return ad


def run_nest_preprocess(adata, mode='all'):
    ligand_dict_dataset, l_r_pair = get_annotations(adata)
    cell_percentile = calculate_cell_percentiles(adata)
    dist_X = adata.obsp['knn_adj'].toarray()
    cells_ligand_vs_receptor = get_cell(adata, ligand_dict_dataset, cell_percentile, l_r_pair, dist_X)
    if mode == 'all':
        row_col, edge_weight, lig_rec, self_loop_found = get_attrs_all(cells_ligand_vs_receptor, dist_X,
                                                                       ligand_dict_dataset)
    else:
        row_col, edge_weight, lig_rec, self_loop_found = get_attrs(cells_ligand_vs_receptor, dist_X)
    adata.uns['edge_list'] = row_col
    adata.uns['edge_weight'] = edge_weight
    adata.uns['lig_rec'] = lig_rec
    # Convert defaultdict to regular dict for h5ad serialization
    adata.uns['self_loop_found'] = dict(self_loop_found)
    return adata


def get_annotations(adata):
    print('ligand-receptor database reading done.')
    print('Preprocess start.')
    from collections import defaultdict
    df = pd.read_csv(str(PROJECT_ROOT / "configs/lib/CellNEST_database.csv"), sep=",")
    gene_info = dict.fromkeys(adata.var.index.astype(str).str.upper(), '')
    ligand_dict_dataset, cell_cell_contact = defaultdict(list), {}
    valid_pairs = df[df['Ligand'].isin(gene_info) & df['Receptor'].isin(gene_info)]
    for _, row in valid_pairs.iterrows():
        ligand, receptor, annotation = row['Ligand'], row['Receptor'], row['Annotation']
        ligand_dict_dataset[ligand].append(receptor)
        gene_info[ligand] = gene_info[receptor] = 'included'
        if annotation == 'Cell-Cell Contact':
            cell_cell_contact[receptor] = ''
    print(f'Number of ligands: {len(ligand_dict_dataset)}')
    gene_ids = adata.var.index.astype(str).str.upper().tolist()
    included_gene = [gene for gene in gene_info if gene_info[gene] == 'included']
    print(f'Total genes: {len(gene_ids)}, ligand/receptor genes: {len(included_gene)}')
    l_r_pair, lr_id = {}, 0
    for ligand, receptors in ligand_dict_dataset.items():
        unique_receptors = list(set(receptors))
        ligand_dict_dataset[ligand] = unique_receptors
        l_r_pair[ligand] = {receptor: lr_id + i for i, receptor in enumerate(unique_receptors)}
        lr_id += len(unique_receptors)
    print(f'Number of ligand-receptor pairs: {lr_id}')
    return ligand_dict_dataset, l_r_pair


def calculate_cell_percentiles(adata, threshold=98):
    cell_vs_gene = np.array(adata.X)
    cell_percentile = []
    for i in range(cell_vs_gene.shape[0]):
        y = sorted(cell_vs_gene[i])
        active_cutoff = np.percentile(y, threshold)
        if active_cutoff == min(cell_vs_gene[i][:]):
            active_cutoff = max(cell_vs_gene[i][:])
        cell_percentile.append(active_cutoff)
    return cell_percentile


def get_cell(adata, ligand_dict_dataset, cell_percentile, l_r_pair, dist_X):
    gene_index = {gene: i for i, gene in enumerate(adata.var.index.astype(str).str.upper())}
    cell_vs_gene = np.array(adata.X)
    n_cells = cell_vs_gene.shape[0]
    ligand_list = list(ligand_dict_dataset.keys())
    cells_ligand_vs_receptor = [[[] for _ in range(n_cells)] for _ in range(n_cells)]
    count_total_edges = 0
    for g, gene in tqdm(enumerate(ligand_list), total=len(ligand_list), desc='Processing ligand genes'):
        ligand_idx = gene_index[gene]
        ligand_active = cell_vs_gene[:, ligand_idx] >= cell_percentile
        for gene_rec in ligand_dict_dataset[gene]:
            receptor_idx = gene_index[gene_rec]
            receptor_active = cell_vs_gene[:, receptor_idx] >= cell_percentile
            relation_id = l_r_pair[gene][gene_rec]
            valid_pairs = np.logical_and(ligand_active[:, None], receptor_active)
            valid_pairs = np.logical_and(valid_pairs, dist_X > 0)
            scores = cell_vs_gene[:, ligand_idx][:, None] * cell_vs_gene[:, receptor_idx]
            rows, cols = np.where(valid_pairs)
            for i, j in zip(rows, cols):
                score = scores[i, j]
                if score <= 0:
                    print('Zero valued ccc score found. Might be a potential ERROR!!')
                    continue
                cells_ligand_vs_receptor[i][j].append([gene, gene_rec, score, relation_id])
                count_total_edges += 1
    return cells_ligand_vs_receptor


def get_attrs(cells_ligand_vs_receptor, dist_X):
    # Convert inputs to NumPy arrays for faster operations
    dist_X = np.asarray(dist_X)
    # Pre-allocate lists with approximate sizes to avoid dynamic resizing
    row_col, edge_weight, lig_rec = [], [], []
    self_loop_found = defaultdict(dict)
    valid_indices = np.nonzero(dist_X > 0)
    for i, j in zip(valid_indices[0], valid_indices[1]):
        interactions = cells_ligand_vs_receptor[i][j]
        if interactions:  # Check if the interaction list is non-empty
            for gene, gene_rec, score, extra in interactions:
                row_col.append([i, j])
                edge_weight.append([dist_X[i, j], score, extra])
                lig_rec.append([gene, gene_rec])
                if i == j:
                    self_loop_found[i][j] = ''
    return row_col, edge_weight, lig_rec, self_loop_found


def get_attrs_all(cells_ligand_vs_receptor, dist_X, ligand_dict_dataset):
    random.seed(42)
    # Convert inputs to NumPy arrays for faster operations
    dist_X = np.asarray(dist_X)
    # Pre-allocate lists with approximate sizes to avoid dynamic resizing
    row_col, edge_weight, lig_rec = [], [], []
    self_loop_found = defaultdict(dict)
    valid_indices = np.nonzero(dist_X > 0)
    for i, j in zip(valid_indices[0], valid_indices[1]):
        tmp = ligand_dict_dataset.copy()
        interactions = cells_ligand_vs_receptor[i][j]
        if interactions:  # Check if the interaction list is non-empty
            for gene, gene_rec, score, extra in interactions:
                row_col.append([i, j])
                edge_weight.append([dist_X[i, j], score, extra])
                lig_rec.append([gene, gene_rec])
                if i == j:
                    self_loop_found[i][j] = ''
                if gene in tmp:
                    del tmp[gene]
        for ligand in random.sample(list(tmp.keys()), 5):
            for receptor in tmp[ligand]:
                row_col.append([i, j])
                edge_weight.append([dist_X[i, j], 0, 0])
                lig_rec.append([ligand, receptor])
                if i == j:
                    self_loop_found[i][j] = ''
    return row_col, edge_weight, lig_rec, self_loop_found
