"""Manuscript Figure 4: spatial response, enrichment and regulatory support."""
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .common import read,combine,COLORS
from .networks import cell_type_map,spatial_axes,MARKER_CMAP,sankey,compact_sankey
from .figure2 import curves

def panel_a():
    spots=read('marker_strata_all_spots.csv')
    return cell_type_map(spots[spots.drug.eq('control')])

def panel_b():
    spots=read('marker_strata_all_spots.csv');control=spots[spots.drug.eq('control')]
    r=read('regional_response.csv').set_index('stratum').loc[['proximal','distal']]
    fig=plt.figure(figsize=(13.3,3.85));axes=[fig.add_axes(v) for v in [(.015,.16,.245,.70),(.325,.16,.245,.70),(.66,.16,.33,.78)]]
    im=axes[0].scatter(control.x,control.y,c=control.plaque_score,s=3,linewidths=0,cmap=MARKER_CMAP,vmin=spots.plaque_score.min(),vmax=spots.plaque_score.max(),rasterized=True)
    cb=fig.colorbar(im,cax=fig.add_axes([.27,.30,.009,.42]));cb.set_label('Marker score',fontsize=10)
    for key,label,color in [('distal','Low','#b3cde3'),('intermediate','Middle','#dce1e3'),('proximal','High','#fbb4ae')]:
        d=control[control.stratum.eq(key)];axes[1].scatter(d.x,d.y,color=color,s=3,linewidths=0,label=label,rasterized=True)
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,markerscale=3,frameon=False)
    for ax in axes[:2]:spatial_axes(ax,control)
    for offset,col,label,color in [(-.12,'spearman_pred_vs_obs','Change correlation (ρ)','#7e9faf'),(.12,'frac_specific_down_reproduced','Down-gene recovery','#e0bd95')]:
        bars=axes[2].bar(np.arange(2)+offset,r[col],.20,yerr=r[col+'_sd'],capsize=3,color=color,label=label,edgecolor='#555',lw=.6)
        axes[2].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[2].set(xticks=[0,1],xticklabels=['High score','Low score'],xlim=(-.5,1.5),ylim=(0,1.18),ylabel='Correlation / recovered fraction')
    axes[2].legend(loc='upper center',fontsize=8,frameon=False)
    return fig

def enrichment(panel):
    if panel=='c':
        d=read('module_enrichment.csv');d=d[d.Method.eq('spike')].sort_values('Mod_Num');vals=d.Log_P;labels=d.Annotation;title='Functional-module enrichment'
    else:
        d=read('go_enrichment.csv');vals=-np.log10(d['P-value']);labels=d.Term;title='GO/pathway enrichment'
    fig,ax=plt.subplots(figsize=(9.8,2.68));ax.barh(range(len(d)),vals,color=COLORS['spike'],edgecolor='#777',lw=.5)
    ax.set(yticks=range(len(d)),yticklabels=['\n'.join(textwrap.wrap(t,57)) for t in labels],xlabel='−log10(unadjusted enrichment p)')
    if panel=='c':
        xmax=vals.max()
        for i,row in enumerate(d.itertuples()):ax.text(vals.iloc[i]+.018*xmax,i,f'{int(row.Overlap_Count)}/{int(row.Cluster_Size)}',ha='left',va='center',fontsize=9)
        ax.set_xlim(0,xmax*1.25);ax.tick_params(axis='y',labelsize=9);ax.tick_params(axis='x',labelsize=9)
    else:ax.tick_params(labelsize=7);ax.xaxis.label.set_size(8)
    ax.set_title(title,fontsize=13 if panel=='c' else 12,weight='bold',pad=7);ax.invert_yaxis();fig.tight_layout(pad=.5)
    return fig

def panel_c():return enrichment('c')
def panel_d():return enrichment('d')

def network(method):
    full=read(f'rt_{method}.csv')
    receptors=full.groupby('receptor')['mean'].max().nlargest(12).index
    tfs=full.groupby('tf')['mean'].max().nlargest(12).index
    d=full[full.receptor.isin(receptors)&full.tf.isin(tfs)].sort_values(['mean','receptor','tf'],ascending=[False,True,True]).head(40).copy()
    lo,hi=full['mean'].min(),full['mean'].max()
    d['normalized']=(d['mean']-lo)/(hi-lo) if hi>lo else 0.
    receptors=sorted(d.receptor.unique());tfs=sorted(d.tf.unique())
    fig,ax=plt.subplots(figsize=(6,4));cmap=LinearSegmentedColormap.from_list('support',['#F1FAFA','#3898B0','#0F2651'])
    ax.scatter(d.tf.map({v:i for i,v in enumerate(tfs)}),d.receptor.map({v:i for i,v in enumerate(receptors)}),c=d.normalized,s=20+110*d.normalized,cmap=cmap,vmin=0,vmax=1,edgecolors='none')
    ax.set(xticks=range(len(tfs)),xticklabels=tfs,yticks=range(len(receptors)),yticklabels=receptors,xlabel='Transcription Factors',ylabel='Receptors')
    ax.tick_params(axis='x',rotation=45,labelsize=11,length=0);ax.tick_params(axis='y',labelsize=11,length=0)
    ax.invert_yaxis()
    for spine in ax.spines.values():spine.set_visible(False)
    fig.tight_layout(pad=.4,rect=(0,0,1,.91));fig.text(.5,.985,method,ha='center',va='top',fontsize=14,weight='bold')
    return fig

def panel_e():
    methods=['spike','state','scGPT','scGen','CellOT'];parts={m:network(m) for m in methods}
    result=combine(parts,{m:((i%3)/3,(i//3)/2,1/3,1/2) for i,m in enumerate(methods)},(18,8),False)
    for f in parts.values():plt.close(f)
    return result

def cascade(method):
    rt=read(f'{method}_tf_links.csv');tg=read(f'{method}_tg_links.csv');rt,tg=compact_sankey(rt,tg)
    fig,ax=plt.subplots(figsize=(7.8,4.4));sankey(ax,rt,tg,label_fontsize=11,label_spacing=.048)
    fig.subplots_adjust(left=.03,right=.945,bottom=.01,top=.86);fig.text(.5,.985,method.title(),ha='center',va='top',fontsize=14,weight='bold')
    return fig

def panel_f():
    parts={m:cascade(m) for m in ['spike','state']}
    result=combine(parts,dict(spike=(0,0,1,.5),state=(0,.5,1,.5)),(7.8,8.8),False)
    for f in parts.values():plt.close(f)
    return result

def panel_g():
    data=read('reprogramming_distributions.csv');data=data[data.Gene.eq('Ttr')]
    fig,ax=plt.subplots(figsize=(6,6*265/420));fig.subplots_adjust(left=.12,right=.975,bottom=.17,top=.79)
    for m,(x,y) in curves(data,['Control','Treated','spike']).items():
        ax.plot(x,y,color={'Control':'#808080','Treated':'#0000ff','spike':'#7fabd0'}[m],lw=2,label={'Control':'Control','Treated':'Observed treated','spike':'Spike'}[m])
    ax.text(.5,1.06,'Ttr',transform=ax.transAxes,ha='center',fontstyle='italic',fontsize=12)
    ax.set(xlim=(-.02*x[-1],x[-1]),ylim=(-.02,1.05),yticks=[0,.5,1],xticks=[0,5,10],ylabel='Relative density')
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.54,1),ncol=3,frameon=False,fontsize=9.5,handlelength=1.5,columnspacing=.8,handletextpad=.4)
    fig.supxlabel('Expression (log1p CPM10k)',y=.025,fontsize=10.5)
    return fig

PANELS=dict(a=panel_a,b=panel_b,c=panel_c,d=panel_d,e=panel_e,f=panel_f,g=panel_g)
def complete(figures):
    # G occupies the unused lower-right slot of E's 2x3 network arrangement.
    return combine(figures,dict(a=(.025,.04,.285,.25),b=(.335,.04,.64,.25),c=(.025,.35,.475,.17),d=(.53,.35,.445,.17),e=(.025,.59,.67,.40),f=(.72,.59,.255,.40),g=(.47,.80,.225,.185)),(21,14.6))
