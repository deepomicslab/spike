# Reproduce the manuscript figures

These notebooks **draw from numerical data by default**. Each panel has its own executable cell and matching caption. After the panels, a final cell assembles the complete figure and checks that all outputs were produced. No `RENDER` switch or manuscript PNG is required.

## Install and run

From the repository root, use the Python 3.12 environment in the [installation instructions](../README.md#installation--setup):

```bash
python -m pip install -r requirements-figures.txt
jupyter lab notebooks/
```

Choose that environment's kernel, then **Restart Kernel and Run All Cells**. The 33 checked input tables are already included in [`reproduction/tables/`](../reproduction/tables/). No external dataset folder, model weights or GPU is required.

| Notebook | Paper panels | Plotting implementation |
| --- | --- | --- |
| [figure2.ipynb](figure2.ipynb) | A–E: expression, response, spatial maps, distributions, TF support | [figure2.py](../reproduction/figure2.py) |
| [figure3.ipynb](figure3.ipynb) | A–F: KEGG, LR changes, NCAM1 expression/network, two catalogue comparisons | [figure3.py](../reproduction/figure3.py) |
| [figure4.ipynb](figure4.ipynb) | A–G: cell types, marker groups, enrichment, regulatory networks/cascades, Ttr | [figure4.py](../reproduction/figure4.py) |
| [figure5.ipynb](figure5.ipynb) | A–E: historical architecture ablation and manuscript gene-count results | [figure5.py](../reproduction/figure5.py) |

Figure 1 is the conceptual overview shown in the main README. It has no statistical reproduction notebook.

## Execute the collection

```bash
python scripts/execute_figure_notebooks.py
# Or execute one figure:
python scripts/execute_figure_notebooks.py --figure 4
```

The runner starts a clean kernel for each notebook and stops on the first error. After successful execution, it saves all cell outputs and inline figure previews directly into the original notebook in `notebooks/`, and keeps a copy in `outputs/executed_notebooks/`. A failed execution leaves the original notebook intact.

Each figure produces `outputs/reproduction/figureN/a.png` (and PDF/SVG), subsequent panels, `complete.png`/PDF/SVG, and `validation.json`. The latter records input-manifest, plotting-code and output hashes. `PASS` confirms successful redraw and checked inputs; it does not mean binary identity with the reference PDF across graphics-library versions.

The notebooks include the visualizations from their latest successful execution, so you can inspect each panel and the complete figure directly in Jupyter or GitHub's notebook preview. Re-running all cells recalculates the plots from the bundled data tables and refreshes these outputs. Images in `manuscript/` serve as reference illustrations.

Optional environment variables:

- `SPIKE_CODE_ROOT`: repository path when running outside it.
- `SPIKE_FIGURE_DATA`: directory containing the checked `manifest.json` and `tables/` (default: `reproduction/`).
- `SPIKE_FIGURE_OUTPUT`: location for regenerated figures (default: `outputs/reproduction/`).

## Which results are reproduced?

The notebook numbers are **manuscript numbers**, corresponding to the 13 October 2026 PDF. Historical plotting-package IDs 1–4 correspond to manuscript Figures 2–5; they are not used as notebook labels.

The manuscript uses released graph Spike predictions across all seven datasets and matched split-42 State results, seeds 41/42/43. This is explicitly separate from the original frozen graph/expression-only selection. Figure 5A–D retain their historical architecture results; E uses conditional-residual results. The original model source, frozen choices and active pointer are unchanged.

See the [table provenance and validation guide](../reproduction/README.md). These notebooks reproduce figures from released numerical results, not model training.
