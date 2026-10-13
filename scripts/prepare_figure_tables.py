"""Maintainer-only extraction of released numerical data; never fits models."""
from pathlib import Path
import json,hashlib,shutil,argparse
import numpy as np,pandas as pd,h5py
parser=argparse.ArgumentParser(description="Extract plot-ready values from released bundles; requires h5py.")
parser.add_argument('--source-root',required=True,type=Path)
parser.add_argument('--manuscript-root',required=True,type=Path)
parser.add_argument('--destination',type=Path,default=Path(__file__).resolve().parents[1])
args=parser.parse_args()
ROOT=args.source_root.resolve()
STAGE=args.destination.resolve()
TABLES=STAGE/'reproduction/tables'
TABLES.mkdir(parents=True,exist_ok=True)
SOURCES={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):
 p=Path(p); SOURCES[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return pd.read_csv(source(p))
def save(df,name):df.to_csv(TABLES/name,index=False,float_format='%.17g')
def copy(p,name=None):p=source(p);shutil.copy2(p,TABLES/(name or p.name))
base=ROOT/'experiments/latest_spike_figures_20260920'
copy(ROOT/'experiments/tf_cosine_manuscript_update_20261009/figure2b_source.csv','benchmark.csv')
for name in ['kegg_PSAPP','kegg_highlights_PSAPP','lr_biological_annotations_PSAPP','spatial_spots_PSAPP','ncam1_celltype_spots_PSAPP','ncam1_celltype_edges_PSAPP','ncam1_celltype_routes_PSAPP','catalog_top20_hits_PSAPP']:
 copy(base/f'figure2/{name}.csv')
for name in ['marker_strata_all_spots','regional_response','module_enrichment','go_enrichment','spike_tf_links','spike_tg_links','state_tf_links','state_tg_links','reprogramming_distributions']:
 copy(base/f'figure3/{name}.csv')
for m in ['spike','state','scGPT','scGen','CellOT']:
 d=read(base/f'figure3/rt_{m}.csv')
 if 'score' in d:d=d.groupby(['receptor','tf'],sort=True).score.mean().rename('mean').reset_index()
 save(d[['receptor','tf','mean']],f'rt_{m}.csv')
for name in ['fig4a_architecture_ablation_per_seed','fig4a_architecture_paired_wilcoxon','fig4b_architecture_means','original_spatial_values','spatial_plot_coordinates','fig4e_lr_candidates','fig4e_lr_summary']:
 copy(base/f'figure4/{name}.csv')
# Published State expression exports, aligned by explicit cell/gene identifiers.
sroot=ROOT/'state_matched_split42_20261007/results'
def state_columns(ds,genes):
 frames=[]
 for seed in [41,42,43]:
  p=(sroot/f'figure1_seed42_contract/{ds}/state/seed_42/imputed.h5ad' if seed==42 else sroot/f'state_matched_completion_20261007/gene5000/seed_{seed}/{ds}/model-state/figure1a_test_predictions/imputed.h5ad')
  source(p)
  with h5py.File(p) as h:
   ids=pd.Index(h['obs']['_index'].asstr()[:]); names=pd.Index(h['var']['_index'].asstr()[:]); x=h['X'][:].astype(float)
   assert np.isfinite(x).all()
   # Canonical released log1p predictions: restore counts, clip and renormalize.
   raw=np.maximum(np.expm1(x),0); total=raw.sum(1,keepdims=True);x=np.log1p(np.divide(raw*10000,total,out=np.zeros_like(raw),where=total>0))
   frames.append(pd.DataFrame(x[:,names.get_indexer(genes)],index=ids,columns=genes))
 for f in frames: assert f.index.equals(frames[0].index)
 return sum(frames)/3
spatial=read(base/'figure1/spatial_expression.csv');graph=read(ROOT/'spike_fixed_graph_sync_20261005/tables/brain_sleep3_Ttr_spatial.csv').set_index('spot_id')
spatial.loc[spatial.method.eq('Spike'),'expression']=spatial.loc[spatial.method.eq('Spike'),'index'].map(graph.graph_mean)
state=state_columns('brain_sleep3',['Ttr'])
# Retain the six source-population maps; replace the outdated State export by spot ID.
state_map=spatial[spatial.method.eq('Source')].copy();state_map['method']='State';state_map['expression']=state_map['index'].map(state.Ttr)
spatial=pd.concat([spatial[spatial.method.ne('State')],state_map],ignore_index=True)
assert spatial.expression.notna().all();save(spatial,'benchmark_spatial.csv')
expr=read(base/'figure1/expression_distributions_with_state_returned.csv')
expr=expr[~expr.Source.str.lower().str.startswith('state')].copy(); state=state_columns('brain_sleep1',['Sgk1','Ttr'])
rows=[dict(Expression=v,Source='state',Dataset='brain_sleep1',Gene=g,Seed='41;42;43',Cell=c,split_seed=42) for g in state for c,v in state[g].items()]
expr=pd.concat([expr,pd.DataFrame(rows)],ignore_index=True);save(expr,'benchmark_distributions.csv')
# Gene-count curves: published graph runs and matched State metrics; no fitting/selection.
d=read(base/'figure4/fig4d_method_comparison_per_seed.csv')
d=d[d.method.ne('spike')][['dataset','gene_length','method','seed','Pearson','RMSE']]
graph=read(ROOT/'spike_fixed_graph_sync_20261005/tables/graph_all_counts_per_seed.csv');graph['method']='spike'
d=pd.concat([d,graph[d.columns]],ignore_index=True)
rows=[]
for ds in ['brain_sleep1','brain_sleep2','brain_sleep3','brain_sleep4','WT','SHH','PSAPP']:
 for count in [1000,2000,5000,10000]:
  for seed in [41,42,43]:
   if count==5000 and seed==42:
    p=source(sroot/f'figure1_seed42_contract/{ds}/state/seed_42/manifest.json');m=json.loads(p.read_text());pr=m['metrics']['R2'];rm=m['metrics']['RMSE']
   else:
    p=source(sroot/f'state_matched_completion_20261007/gene{count}/seed_{seed}/{ds}/model-state/figure1a_metrics.json');m=json.loads(p.read_text());pr=m['feature_mean_pearson'];rm=m['feature_mean_rmse']
   rows.append(dict(dataset=ds,gene_length=count,method='state',seed=seed,Pearson=pr,RMSE=rm))
d=pd.concat([d,pd.DataFrame(rows)],ignore_index=True);d['method']=d.method.replace({'scgen':'scGen','scgpt':'scGPT','cellot':'CellOT'});assert len(d)==420 and not d.duplicated(['dataset','gene_length','method','seed']).any();save(d,'gene_count_per_seed.csv')
# Record paper identity without bundling or altering its manuscript.
paper=args.manuscript_root.resolve()
reference={'paper_title':'Spike enables spatially informed and biologically interpretable analysis of cellular reprogramming','reference_date':'2026-10-13','manuscript_pdf_sha256':sha(paper/'sn-article.pdf'),'figure_images':{str(i+1):sha(paper/f'Figures/Figure{i}.png') for i in range(1,5)},'protocol':'Released manuscript graph predictions for all seven datasets; matched split-42 State, seeds 41/42/43. Original frozen model selection remains unchanged.','source_files':SOURCES,'tables':{p.name:sha(p) for p in sorted(TABLES.glob('*.csv'))}}
(STAGE/'reproduction/manifest.json').write_text(json.dumps(reference,indent=2)+'\n')
print('Exported',len(reference['tables']),'tables',sum(p.stat().st_size for p in TABLES.glob('*')),'bytes')
