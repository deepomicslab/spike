"""Manuscript Figure 5: historical A-D ablations and current manuscript gene counts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from .common import read,combine,DATASETS,COLORS
ORIGINAL=Path('.')
OUT=Path('.')
ORDER=DATASETS
ARCHITECTURE_VARIANTS=('spike','single_decoder','standard_gat','no_edge')
VARIANTS=ARCHITECTURE_VARIANTS
LABELS=('Full model','Single decoder','Standard GAT','No edge features')
ABLATION_COLORS=dict(spike=COLORS['spike'],single_decoder=COLORS['scGen'],standard_gat=COLORS['state'],no_edge=COLORS['scGPT'])
LAYOUT={'a':(.24,.18,8.95,2.50),'b':(9.58,.18,4.2,2.50),'c':(.24,2.91,5.45,4.60),'d':(5.90,2.91,4.2,4.60),'e':(10.38,2.91,3.4,4.60)}
INK='#555555';GRID='#eeeeee';DATASET_POINT_COLOR='#808080'
SPATIAL_CMAP=LinearSegmentedColormap.from_list('architecture',['#286cbc','#faf5f4','#a8393d'])
SPATIAL_CONDITIONS=dict(zip(VARIANTS,['Full model (Spike)','Single decoder','Standard GAT','No edge features']))
class Context:
    def read(self,path):return read(Path(path).name)
    def table(self,frame,name):return frame

def fixed_save(context,fig,*args,**kwargs):context.figure=fig

def spatial_image_coordinates(spatial):
    coordinates=read('spatial_plot_coordinates.csv')
    assert set(coordinates.spot_id)==set(spatial.spot_id)
    return coordinates,None

def plot_a(f):
    """Reference boxplot geometry with opaque original-color boxes and points."""
    from scipy.stats import wilcoxon
    values = f.read(ORIGINAL/'fig4a_architecture_ablation_per_seed.csv')
    values = values[values.variant.isin(ARCHITECTURE_VARIANTS)]
    means = values.groupby(['dataset', 'variant'])[['Pearson', 'RMSE']].mean()
    fig, axes = plt.subplots(1, 2, figsize=LAYOUT['a'][2:], sharey=True)
    fig.subplots_adjust(left=.17, right=.99, bottom=.20, top=.96, wspace=.22)
    jitter = np.array([-.075, .02, .085, -.01, .06, -.06, .035])
    comparisons = []
    for ax, metric in zip(axes, ('Pearson', 'RMSE')):
        for row, variant in enumerate(ARCHITECTURE_VARIANTS):
            data = np.array([means.loc[(ds, variant), metric] for ds in ORDER])
            ax.bxp([boxplot_stats(data, whis=1.5)[0]], positions=[row], widths=.60,
                   orientation="horizontal", showfliers=False, patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor=ABLATION_COLORS[variant], alpha=1,
                                 edgecolor='#999999', linewidth=.75),
                   whiskerprops=dict(color='#999999', linewidth=.7),
                   capprops=dict(color='#999999', linewidth=.7),
                   medianprops=dict(color='#999999', linewidth=.8))
            ax.scatter(data, row+jitter, s=17, c=ABLATION_COLORS[variant],
                       edgecolor='#777777', linewidth=.25, alpha=1, zorder=3)
            ax.vlines(data.mean(), row-.30, row+.30, color='#333333', linewidth=1, zorder=4)
        ax.set_ylim(3.6, -.6)
        ax.set_yticks(range(4))
        ax.tick_params(axis='both', length=2.5, direction='in', top=True, right=True)
        ax.grid(False)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#999999')
            spine.set_linewidth(.6)
        # Paired comparisons use the seven dataset means, not 21 training seeds.
        full = np.array([means.loc[(ds, 'spike'), metric] for ds in ORDER])
        raw = [wilcoxon(full, np.array([means.loc[(ds, variant), metric] for ds in ORDER]),
                        alternative='two-sided', method='exact').pvalue
               for variant in ARCHITECTURE_VARIANTS[1:]]
        order = np.argsort(raw)
        adjusted = np.empty(3)
        adjusted[order] = np.minimum(1, np.maximum.accumulate(np.array(raw)[order] * np.arange(3, 0, -1)))
        lo, hi = ((.87, 1.007) if metric == 'Pearson' else (.07, .58))
        span = hi - lo
        ax.set_xlim(lo, hi + .32 * span)
        for row, variant in enumerate(ARCHITECTURE_VARIANTS[1:], start=1):
            x = hi + (.025 + (row - 1) * .10) * span
            cap = .022 * span
            ax.plot([x, x+cap, x+cap, x], [0, 0, row, row], color=INK, lw=.65)
            ax.text(x+cap+.017*span, row/2, f'{adjusted[row-1]:.3f}',
                    rotation=270, ha='center', va='center', fontsize=7)
            comparisons.append(dict(metric=metric, reference='spike', variant=variant,
                n_dataset_pairs=len(ORDER), alternative='two-sided', method='exact',
                p_value=raw[row-1], p_holm=adjusted[row-1], correction_family='three comparisons within metric'))
    axes[0].set_yticklabels(['Full model', 'Single decoder', 'Standard GAT',
                            'No edge features'])
    axes[0].set(xticks=[.88, .92, .96, 1.0], xlabel='Feature-mean Pearson r ↑')
    axes[1].set(xticks=[.1, .2, .3, .4, .5], xlabel='Feature-mean RMSE ↓')
    f.table(pd.DataFrame(comparisons), 'fig4a_architecture_paired_wilcoxon')
    caption = ('Architecture ablation: Full model (zero-time Spike), Single decoder, Standard GAT, '
               'and No edge features. Each point is one dataset mean over seeds 41/42/43; '
               'boxes summarize seven dataset means in the fixed 5,000-gene space. '
               'Feature-mean Pearson correlation and RMSE are shown. Reference-inspired opaque colored boxes '
               'show median (gray), IQR and 1.5-IQR whiskers; black vertical segments mark arithmetic means. '
               'Dots retain the blue/purple/green/orange condition palette. Right-side brackets compare '
               'Full model with each variant using exact two-sided paired Wilcoxon signed-rank tests on '
               'seven dataset means; displayed p-values are Holm-adjusted over three comparisons per metric. '
               'These are descriptive comparisons on previously inspected datasets, not independent '
               'biological-replicate inference. Historical loss comparisons are omitted.')
    fixed_save(f, fig, 'a', caption, 'Architecture ablation', context_key='a_architecture')

def plot_c(f, spatial):
    """Four labeled architecture maps on a shared physical coordinate system."""
    coordinates, source = spatial_image_coordinates(spatial)
    f.table(coordinates, 'spatial_plot_coordinates')
    plotted = spatial.merge(coordinates, on='spot_id', validate='many_to_one')
    fig = plt.figure(figsize=LAYOUT['c'][2:])
    low, high = np.percentile(spatial.expression, [2, 98])
    xy = coordinates[['image_x', 'image_y']].to_numpy()
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    pad = np.maximum(upper-lower, 1)*.025
    # Place the equal-aspect maps by their physical size; GridSpec leaves large
    # horizontal gutters when the tissue is narrower than its allotted slot.
    width, height = LAYOUT['c'][2:]
    map_height = 1.84
    map_width = map_height * (upper[0]-lower[0]+2*pad[0]) / (upper[1]-lower[1]+2*pad[1])
    gap_x, gap_y = .18, .28
    left = (.82*width - (2*map_width + gap_x)) / 2
    top = .39
    for i, (method, label) in enumerate(SPATIAL_CONDITIONS.items()):
        x = left + (i % 2) * (map_width + gap_x)
        y = top + (i // 2) * (map_height + gap_y)
        ax = fig.add_axes([x/width, 1-(y+map_height)/height, map_width/width, map_height/height])
        group = plotted[plotted.method.eq(method)]
        im = ax.scatter(group.image_x, group.image_y, c=group.expression, cmap=SPATIAL_CMAP,
                        s=8, linewidths=0, vmin=low, vmax=high)
        ax.set(xlim=(lower[0]-pad[0], upper[0]+pad[0]),
               ylim=(upper[1]+pad[1], lower[1]-pad[1]), aspect='equal')
        # Condition identifiers are necessary to read the maps. Keep them as
        # axis annotations so caption-only manuscript titles do not erase them.
        ax.text(.5, 1.035, label, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333')
        ax.axis('off')
    colorbar_x = (left + 2*map_width + gap_x + .22) / width
    cb = fig.colorbar(im, cax=fig.add_axes([colorbar_x, .25, .014, .50]), extend='both')
    cb.set_label('Predicted Map1b expression', fontsize=10, labelpad=7)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(.5)
    caption = ('SHH / Map1b spatial predictions on the same 880 source spots, averaged over training seeds 41/42/43. '
               'Maps show Full model (zero-time Spike), Single decoder, Standard GAT and No edge features. '
               'All four maps share the reference Figure 4C blue–white–red palette and pooled 2nd/98th-percentile color limits; white denotes the midpoint of the absolute expression range, not zero. '
               'colorbar triangles denote clipped tails. Original architecture predictions are unchanged. '
               'Coordinates are matched by spot ID to SHH adata.obsm[spatial], with image y '
               'increasing downward and equal physical x/y scale. Original array row/column '
               'indices are retained in the source table; they are not isotropic image coordinates. '
               'The example was selected using observed training-response variation; it does not establish causal advantage.')
    fixed_save(f, fig, 'c', caption, 'Architecture spatial predictions', context_key='c_architecture')

def panel_a():
    context=Context();plot_a(context);return context.figure

def panel_b():
    means=read('fig4b_architecture_means.csv')
    matrix=means.pivot(index='dataset',columns='variant',values='distribution_pair_rmse').reindex(index=DATASETS,columns=VARIANTS)
    with sns.axes_style('whitegrid'):
        fig,ax=plt.subplots(figsize=(4.2,2.5))
        sns.violinplot(data=matrix,bw_adjust=.5,cut=1,linewidth=1,palette=[ABLATION_COLORS[v] for v in VARIANTS],saturation=1,inner='box',ax=ax)
        offsets=np.array([-.09,.035,.10,-.035,.07,-.10,0.])
        for i,v in enumerate(VARIANTS):ax.scatter(i+offsets,matrix[v],s=10,color=DATASET_POINT_COLOR,edgecolor='white',linewidth=.3,zorder=4)
        sns.despine(ax=ax,left=True,bottom=True)
    fig.subplots_adjust(left=.19,right=.99,bottom=.20,top=.96)
    ax.set_xticks(range(4),['Full\nmodel','Single\ndecoder','Standard\nGAT','No edge\nfeatures'],fontsize=7)
    ax.set_xlabel('');ax.set_ylabel('Distribution-pair RMSE ↓',fontsize=9,labelpad=5)
    ax.tick_params(axis='both',length=0);ax.tick_params(axis='y',labelsize=8)
    ax.xaxis.grid(False);ax.yaxis.grid(True,color='#e6e6e6',linewidth=.5);ax.set_axisbelow(True)
    return fig

def panel_c():
    context=Context();plot_c(context,read('original_spatial_values.csv'));return context.figure

def prepare_data(context):
    pairs=read('fig4e_lr_candidates.csv');summary=read('fig4e_lr_summary.csv')
    return pairs[pairs.selected].copy(),summary
def plot_d(f):
    pairs, summary = prepare_data(f)
    if not summary.agreement_mean.between(.50, .95).all():
        raise ValueError('LR means exceed the displayed 0.50–0.95 range')
    maxima = summary.groupby('lr_pair').agreement_mean.transform('max')
    winners = summary[np.isclose(summary.agreement_mean, maxima, rtol=0, atol=1e-12)].copy()
    f.table(winners, 'fig4d_lr_best_models')
    fig, ax = plt.subplots(figsize=LAYOUT['d'][2:])
    fig.subplots_adjust(left=.30, right=.955, bottom=.13, top=.81)
    y = np.arange(len(pairs))
    height = .18
    for i, (variant, label) in enumerate(zip(VARIANTS, LABELS)):
        rows = summary[summary.variant.eq(variant)].set_index('lr_pair').loc[pairs.lr_pair]
        ax.barh(y + (i - 1.5) * height, rows.agreement_mean, height=height,
                color=ABLATION_COLORS[variant], label=label)
        if variant == 'spike':
            best = rows.index.isin(winners.loc[winners.variant.eq(variant), 'lr_pair'])
            ax.scatter(rows.agreement_mean.to_numpy()[best] + .008,
                       (y + (i - 1.5) * height)[best], marker='*', s=35,
                       color='#333333', linewidths=.3, zorder=4,
                       gid='best-lr-mean-spike')
    ax.set(yticks=y, yticklabels=pairs.lr_pair, ylim=(len(pairs)-.45, -.6),
           xlim=(.50, .95), xticks=[.50, .60, .70, .80, .90, .95],
           xlabel='LR correlation agreement ↑\n(1 − |Δr| / 2)')
    ax.tick_params(axis='y', labelsize=7, length=0, pad=4)
    ax.tick_params(axis='x', labelsize=8, length=2)
    ax.grid(axis='x', color=GRID, linewidth=.5)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    # Mark the nonzero origin: bar lengths are visually truncated, while the
    # tick labels and all endpoints retain the original agreement values.
    for offset in (-.013, .007):
        ax.plot([offset, offset+.014], [-.009, .009], transform=ax.transAxes,
                color=INK, linewidth=.8, clip_on=False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color='#333333', marker='*', linestyle='none', markersize=6))
    labels.append('Spike best mean')
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(.38, 1.03), ncol=2,
              frameon=False, fontsize=7.2, handlelength=1.4, columnspacing=1)
    caption = (
        'SHH; four historical architecture variants identical to A–C: Full model (zero-time Spike), '
        'Single decoder, Standard GAT and No edge features. Full model here is not the current '
        'conditional-residual model in E. Ten shared database ligand–receptor pairs are ranked '
        'only by observed-target absolute covariance (population SD product times absolute Pearson r); '
        'ties use pair names. All eligible pair scores are retained in the source table. '
        'Each bar is the mean of 1 − |r_pred − r_target| / 2 across seeds 41/42/43; error bars are omitted. '
        'A star is shown only where Full model (historical Spike) has the highest mean '
        'among the four variants for that LR pair, including ties within 1e-12. '
        'Other variants retain their bars without stars; all winners remain in the source table. '
        'Stars denote descriptive rank, not statistical significance. '
        'The horizontal axis is zoomed to 0.50–0.95, with a break mark at its nonzero origin; '
        'all means are shown in full, but visible bar lengths must not be interpreted as ratios from zero. '
        'Training-seed sample SD is retained in the numerical source table. Pearson r measures ligand/receptor '
        'coexpression within the 880 predicted source cells or 616 observed target cells in the same '
        'fixed 5,000-gene CPM10,000/log1p space, split seed 42. The same ten pairs and populations '
        'are used for every variant; undefined predicted correlations are rejected. The prediction '
        'identity is checked against the existing Panel C Map1b values. Target-based pair selection '
        'makes this an exploratory display on previously inspected data; coexpression agreement '
        'does not establish cell–cell communication or causal effects.'
    )
    fixed_save(f, fig, 'd', caption, 'Architecture variants — LR-pair coexpression agreement')

def panel_d():
    context=Context();plot_d(context);return context.figure

def panel_e():
    values=read('gene_count_per_seed.csv')
    dataset_means=values.groupby(['dataset','gene_length','method'])[['Pearson','RMSE']].mean()
    summary=dataset_means.groupby(['method','gene_length']).mean()
    counts=np.array([1000,2000,5000,10000]);fig=plt.figure(figsize=(3.4,4.6))
    grid=fig.add_gridspec(2,1,left=.20,right=.98,bottom=.12,top=.82,hspace=.40)
    palette=dict(spike='#7fabd0',scGen='#b47cc7',scGPT='#ff9900',CellOT='#bbbb00',state='#78b66d')
    markers=dict(spike='o',scGen='s',scGPT='^',CellOT='v',state='D')
    for i,metric in enumerate(['Pearson','RMSE']):
        ax=fig.add_subplot(grid[i,0])
        for m in ['spike','scGen','scGPT','CellOT','state']:
            vals=summary.loc[m].loc[counts,metric]
            ax.plot(counts,vals,color=palette[m],marker=markers[m],lw=1.5,ms=4,label={'spike':'Spike','state':'State'}.get(m,m),markeredgecolor=INK,markeredgewidth=.45,clip_on=False)
        ax.set_xscale('log');ax.set_xticks(counts,['1k','2k','5k','10k']);ax.minorticks_off();ax.set_xlim(850,11800)
        ax.set_ylabel('Expression Pearson r ↑' if i==0 else 'Expression RMSE ↓',fontsize=8)
        ax.tick_params(length=2,pad=3,labelsize=8);ax.grid(axis='y',color=GRID,linewidth=.5)
        if i==0:handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,1),ncol=2,frameon=False,fontsize=8.5)
    fig.supxlabel('Number of genes',x=.59,y=.025,fontsize=9)
    return fig

PANELS=dict(a=panel_a,b=panel_b,c=panel_c,d=panel_d,e=panel_e)
def complete(figures):
    return combine(figures,{k:(x/14,y/7.75,w/14,h/7.75) for k,(x,y,w,h) in LAYOUT.items()},(14,7.75))
