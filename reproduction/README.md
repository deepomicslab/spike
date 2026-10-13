# Numerical figure reproduction

This directory contains the portable drawing code and **33 numerical tables** for manuscript Figures 2–5. Run the [notebooks](../notebooks/README.md) or `python scripts/execute_figure_notebooks.py` from the repository root.

The reference is the manuscript **Spike enables spatially informed and biologically interpretable analysis of cellular reprogramming**, snapshot dated 2026-10-13. `manifest.json` identifies the reference PDF and four manuscript figure images by SHA-256, lists source files relative to the supplied DrugPerturb data package, and records the checksums of the portable CSV files. Those upstream paths are provenance records, not runtime dependencies.

## Sources and display rules

| Paper panel | Input tables | Operation |
| --- | --- | --- |
| 2A/B/E | `benchmark.csv` | Plot the released dataset means; calculate across-dataset mean, sample SD and median guides |
| 2C | `benchmark_spatial.csv` | Draw six maps on fixed coordinates with one expression scale |
| 2D | `benchmark_distributions.csv` | Fit reflected positive-value KDEs, 1.5 × Scott bandwidth; scale each curve to its peak |
| 3A | `kegg_PSAPP.csv`, `kegg_highlights_PSAPP.csv` | Show fixed top-15 terms, sorted by minimum-condition FDR/name; highlight all displayed Target FDR < 0.05 terms |
| 3B | `lr_biological_annotations_PSAPP.csv` | Plot absolute log2 change, direction, ΔM and recorded literature annotations |
| 3C/D | `spatial_spots_PSAPP.csv`, `ncam1_celltype_*` | Draw NCAM1 expression and all 302 directed connections; route percentages use all edges |
| 3E/F | `catalog_top20_hits_PSAPP.csv` | Plot pool-normalized and L×R-null enrichment as distinct panels |
| 4A/B | `marker_strata_all_spots.csv`, `regional_response.csv` | Draw observed annotations, marker groups and released recovery summaries |
| 4C/D | `module_enrichment.csv`, `go_enrichment.csv` | Plot unadjusted enrichment P-values; show module overlap counts |
| 4E | `rt_spike.csv`, `rt_state.csv`, `rt_scGPT.csv`, `rt_scGen.csv`, `rt_CellOT.csv` | Select the recorded network display subset and min–max normalize against each complete aggregate table |
| 4F | `spike_tf_links.csv`, `spike_tg_links.csv`, `state_tf_links.csv`, `state_tg_links.csv` | Draw weighted cascades with common node limits |
| 4G | `reprogramming_distributions.csv` | Draw the brain_sleep1 Ttr positive-expression KDE |
| 5A/B | `fig4a_architecture_ablation_per_seed.csv`, `fig4b_architecture_means.csv` | Historical dataset-level box/violin summaries; recompute paired Wilcoxon/Holm annotations |
| 5C | `original_spatial_values.csv`, `spatial_plot_coordinates.csv` | Historical SHH/Map1b predictions on physical coordinates |
| 5D | `fig4e_lr_candidates.csv`, `fig4e_lr_summary.csv` | Plot historical LR agreement and descriptive Full-model maxima |
| 5E | `gene_count_per_seed.csv` | Average three seeds per dataset, then seven datasets equally; 420 published runs |

The original `kegg_highlights_PSAPP.csv` includes an older six-highlight policy. Its term selection/ranking is retained, but the plotting code deliberately computes the current manuscript's Target-FDR highlight mask. The old `highlighted` column is not the display rule.

## Preparation of portable tables

- `benchmark.csv` is a byte-preserved copy of the released `tf_cosine_manuscript_update_20261009/figure2b_source.csv` containing all 35 dataset/method combinations.
- Spatial and distribution tables retain the released observations and comparator predictions. The Spike brain_sleep3 map uses the graph export. State values are aligned by explicit spot/gene IDs across matched split-42 seeds 41/42/43, converted to nonnegative CPM10,000/log1p and averaged cellwise. The old split-0 State populations are not mixed into those plots.
- The `rt_*.csv` tables retain every scored receptor/TF pair after mean aggregation of released cell-level support. Drawing uses those complete aggregates for normalization, not the selected display subset.
- Gene-count tables combine the released graph scores, existing scGen/scGPT/CellOT metrics and matched State metrics. Method spelling is standardized; no run is trained, selected or dropped.
- Historical ablation tables are preserved and remain separate from current conditional-residual results.

Only the small plot-ready tables are needed at runtime. The full scientific data sources and model fitting are outside the scope of this CPU reproduction collection.

## Verification

`common.audit()` checks every table hash and the packaged model source hash. Each notebook then generates every requested panel and composite and records their hashes under `outputs/reproduction/figureN/validation.json`. Composite assembly reads the new Matplotlib canvases in memory; it never reads the manuscript images. Delete `outputs/` and rerun to rebuild the figures.

The reference figure hashes bind the reproduction target to a specific manuscript snapshot. Fresh graphics may have font, antialiasing or spacing differences; a successful execution does not assert byte-for-byte PDF or PNG identity. `validation_summary.json` records the executed-cell counts, numerical assertions and visual review of this implementation.

Re-run the numerical and execution audit after the notebooks:

```bash
python scripts/validate_figure_reproduction.py
```

Maintainers with the original released data bundles can re-extract the CSVs using [`scripts/prepare_figure_tables.py`](../scripts/prepare_figure_tables.py). This optional preparation step additionally needs `h5py` and explicit `--source-root` / `--manuscript-root` paths. Normal notebook users do not need it.
