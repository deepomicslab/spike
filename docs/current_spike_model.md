# Current published Spike model source

The current publication model is `conditional_residual_v1_validation_selected`.
Its portable source is [`perturb/models/conditional_residual.py`](../perturb/models/conditional_residual.py).
This is the only default Spike model implementation distributed in this library.

```python
from perturb.models.conditional_residual import ConditionalResidual, canonicalize_torch
```

This module requires PyTorch and `torch_geometric.nn.GATv2Conv`, including when
`use_graph=False`: both branches construct the same parameter structure.
Reading the source and checking its hash in the figure notebook needs only Python's standard library.
The module has no dependency on the figure-plotting code. Notebooks use the CPU-only drawing code and checked CSVs in `reproduction/`.

## Interface

```python
model = ConditionalResidual(n_genes, shift, training_edges, n_lr, use_graph, width=32)
prediction, delta = model(x, edge_index, edge_attr)
```

| Argument / output | Meaning |
| --- | --- |
| `n_genes` | Number of expression features, in the frozen gene order. |
| `shift` | Tensor of shape `[n_genes]`: target minus source gene means calculated on training populations. Stored as a model buffer. |
| `training_edges` | Nonempty finite tensor of shape `[training_edges, 3]`; two continuous attributes followed by an integer LR category. Supplies normalization statistics and the seen-category mask. |
| `n_lr` | Size of the fixed LR vocabulary; valid input category IDs are `0` through `n_lr - 1`. |
| `use_graph` | The already selected graph (`True`) or expression-only (`False`) branch. |
| `x` | Float expression tensor of shape `[spots, n_genes]` on the frozen CPM10000/log1p scale, without a pseudotime or placeholder column. |
| `edge_index` | Integer graph indices of shape `[2, edges]`. |
| `edge_attr` | Edge attributes of shape `[edges, 3]`, aligned with `edge_index` and the fixed LR vocabulary. |
| `prediction` | Shape `[spots, n_genes]`, after nonnegative clipping and CPM10000/log1p normalization of `x + shift + delta`. |
| `delta` | The learned residual before output normalization; same shape as `x`. |

Inputs, parameters and buffers must use compatible devices and dtypes. Categories
inside the vocabulary but unseen during training map to the padding embedding;
nonfinite attributes, noninteger categories and out-of-range categories are rejected.
The graph contribution is scaled by `0.5`, and the residual head is zero-initialized.

## Frozen publication choices

At 5,000 genes, graph is selected for WT, PSAPP, RHY 1, RHY 2 and
RHY 4; expression-only is selected for RHY 3 and SHH. Seeds 41/42/43
measure optimization variability. Gene-count experiments have their own frozen
selection records; the 5,000-gene choices must not be applied to them automatically.

RHY 1–4 are manuscript display names for the rhynchophylline mouse brain datasets.
The [data/code identifiers](../README.md#dataset-names) and frozen selection records are unchanged.

The original training objective is population gene-mean MSE + neighborhood contrast
MSE + `0.01 * mean(delta ** 2)`. The loss and training loop are outside this model
module. This source addition does not reconstruct a training pipeline, ship weights,
change the figure publication pointer, or change a frozen model choice. Figure 5A–D
retain their historical architecture-ablation/LR-pair results.

The figure notebooks redraw manuscript Figures 2–5 from the checked numerical tables in
`reproduction/`; manuscript images are reference illustrations only. The manuscript
snapshot uses the separately released graph predictions across seven datasets; this
does not change the original frozen selections listed above. Source availability alone does not reproduce fitted predictions;
checkpoint, preprocessing, split, gene-order and LR-vocabulary contracts are also
needed for that purpose. `scripts/train.py` and `scripts/evaluate.py` serve comparator methods only.
The independent historical architecture variants in `perturb/models/ablation.py`
are retained for the Figure 5A–D analysis.

## Source provenance

[`provenance/current_spike_model.json`](../provenance/current_spike_model.json) records
the source package paths, package-manifest hashes, original and portable source
SHA-256 hashes, and the frozen selection sources. All three supplied source copies
have the same SHA-256, verified against their individual package manifests.

The only portability edit removes this unused import:

```python
from figures.paths import ROOT as _PROJECT_ROOT, source_path
```

The remaining source, parameter names, registered buffers and forward computation
are unchanged. The notebook's source-audit cell verifies the portable hash and
reconstructs the original file in memory to verify its hash as well.
