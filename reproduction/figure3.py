"""Manuscript Figure 3: each A-F panel is drawn from its own numerical source."""
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize,LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .common import read,combine,box
METHODS=['spike','spatialDM','COMMOT','stLearn']
LABELS=['Spike','SpatialDM','COMMOT','stLearn']
CELL_TYPES={'Neuron':'#ccebc5','Astrocyte':'#fbb4ae','Oligodendrocyte':'#decbe4','Microglial Cell':'#b3cde3','Pericyte':'#fed9a6','Schwann Cell':'#ffffcc'}
DISPLAY={'Microglial Cell':'Microglia','Schwann Cell':'Schwann cell'}

def panel_a():
    kegg=read('kegg_PSAPP.csv');highlights=read('kegg_highlights_PSAPP.csv');parts={}
    for method,label in zip(METHODS,LABELS):
        d=kegg[kegg.method.eq(method)];sel=highlights[highlights.method.eq(method)]
        terms=sel.loc[sel.displayed,'Term'].tolist()[::-1];highlighted=set(sel.loc[sel.displayed & sel.target_fdr.lt(.05),'Term'])
        labels=['\n'.join(textwrap.wrap(t,42)) for t in terms];heights=np.array([1+.8*t.count('\n') for t in labels]);pos=np.cumsum(heights)-heights/2
        fig=plt.figure(figsize=(3.8,3.3));ax=fig.add_axes([.67,.16,.15,.77]);norm=Normalize(0,max(1,(-np.log10(d.fdr.clip(lower=1e-300))).max()))
        for x,cond in enumerate(['source','target']):
            g=d[d.condition.eq(cond)].set_index('Term').reindex(terms).dropna(subset=['fdr'])
            im=ax.scatter([x]*len(g),[pos[terms.index(t)] for t in g.index],s=10+g.overlap*5,c=-np.log10(g.fdr.clip(lower=1e-300)),cmap='viridis_r',norm=norm,edgecolors='#777',linewidths=.35)
        ax.set(yticks=pos,yticklabels=labels,xticks=[0,1],xticklabels=['Source','Target'],xlim=(-.45,1.55),ylim=(-.15,heights.sum()+.15))
        ax.tick_params(axis='y',labelsize=7.2,length=0,pad=2);ax.tick_params(axis='x',labelsize=8,rotation=50)
        for t,term in zip(ax.get_yticklabels(),terms):
            if term in highlighted:t.set_color('#B8860B');t.set_weight('bold')
        ax.text(.5,1.025,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=10,fontweight='bold')
        ax.spines[['top','right']].set_visible(True)
        cb=fig.colorbar(im,cax=fig.add_axes([.87,.24,.022,.32]));cb.ax.tick_params(labelsize=6.5,length=1);cb.set_label('−log10(FDR)',fontsize=6.5,labelpad=2)
        for j,n in enumerate([2,5,10]):
            fig.text(.925,.88-j*.062,str(n),fontsize=7)
            fig.add_artist(Line2D([.886],[.884-j*.062],marker='o',markersize=np.sqrt(10+n*5),markerfacecolor='#999',markeredgewidth=.3,transform=fig.transFigure,ls=''))
        parts[method]=fig
    result=combine(parts,{m:(.005+i*.25,.015,.245,.97) for i,m in enumerate(METHODS)},(15.5,3.4),False)
    for f in parts.values():plt.close(f)
    return result

def panel_b():
    bars=read('lr_biological_annotations_PSAPP.csv');fig,axes=plt.subplots(1,4,figsize=(18.7,4.0),sharey=True);ymax=max(bars.abs_log2fc.max()*1.45,.3)
    for ax,method,label in zip(axes,METHODS,LABELS):
        d=bars[bars.method.eq(method)].sort_values('abs_log2fc',ascending=False);x=np.arange(len(d))
        ax.bar(x,d.abs_log2fc,color=np.where(d.log2fc>0,'#fb8072','#80cde1'),edgecolor='black',lw=.8)
        ax.set(xticks=x,xticklabels=d.pair,ylim=(0,ymax));ax.tick_params(axis='x',rotation=40,labelsize=8)
        ax.text(.5,1.06,label,transform=ax.transAxes,ha='center',va='bottom',fontsize=12,fontweight='bold')
        for t in ax.get_xticklabels():t.set_ha('right')
        for i,r in enumerate(d.itertuples()):
            if r.celltype_related:ax.scatter(i-.12,r.abs_log2fc+.035*ymax,marker='*',s=32,color='#486b83',zorder=3)
            if r.mechanism_related:ax.scatter(i+.12,r.abs_log2fc+.035*ymax,marker='D',s=15,color='#a84c93',zorder=3)
            ax.text(i,r.abs_log2fc+.085*ymax,f'ΔM:{r.delta_moran:.2f}',rotation=90,ha='center',va='bottom',fontsize=7.5,weight='bold')
    axes[0].set_ylabel('|Log2 Fold Change|',fontsize=14)
    handles=[Patch(facecolor='#fb8072',edgecolor='black',label='Upregulated (>0)'),Patch(facecolor='#80cde1',edgecolor='black',label='Downregulated (<0)'),Line2D([],[],marker='*',color='#486b83',ls='',label='Cell-type association'),Line2D([],[],marker='D',color='#a84c93',ls='',label='Perturbation-mechanism association')]
    fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False,fontsize=12)
    fig.subplots_adjust(left=.06,right=.995,bottom=.34,top=.76,wspace=.075)
    return fig

def panel_c():
    spots=read('spatial_spots_PSAPP.csv');cmap=LinearSegmentedColormap.from_list('ncam',['#eee','#c5dcea','#b3cde3','#7eabc8','#4d8ab0'])
    fig=plt.figure(figsize=(3.1,5.8));high=spots.expression.quantile(.99)
    for y,d in [(.005,spots[spots.drug.eq('control')]),(.44,spots[spots.drug.ne('control')])]:
        ax=box(fig,.05,y,.90,.425);im=ax.scatter(d.x,d.y,c=d.expression,s=6,cmap=cmap,vmin=0,vmax=high)
        ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off')
    cb=fig.colorbar(im,cax=box(fig,.12,.9,.76,.018),orientation='horizontal');cb.set_label('NCAM1 · log1p CPM10k',fontsize=10,labelpad=4)
    cb.ax.tick_params(labelsize=9,length=3)
    return fig

def panel_d():
    source=read('ncam1_celltype_spots_PSAPP.csv');edges=read('ncam1_celltype_edges_PSAPP.csv');routes=read('ncam1_celltype_routes_PSAPP.csv')
    assert len(edges)==302 and abs(routes.percent_all_edges.sum()-100)<1e-10
    fig=plt.figure(figsize=(9.74,5.8));ax=box(fig,.015,.19,.49,.74)
    for name,c in CELL_TYPES.items():
        d=source[source.cell_type.eq(name)];ax.scatter(d.x,d.y,s=7,color=c,alpha=.6,linewidths=0)
    xy=source.set_index('spot_id')[['x','y']];segments=np.stack([xy.loc[edges.source_id].to_numpy(),xy.loc[edges.target_id].to_numpy()],axis=1)
    ax.add_collection(LineCollection(segments,colors='#46535c',linewidths=.45+1.15*edges.attention/edges.attention.max(),alpha=.55))
    for name,c in CELL_TYPES.items():
        d=source[source.cell_type.eq(name)&source.connected_endpoint];ax.scatter(d.x,d.y,s=23,color=c,edgecolors='#333',lw=.65,zorder=4)
    ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');ax.text(.5,-.055,'Outlined spots: connected endpoints',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    handles=[Line2D([],[],marker='o',ls='',markersize=7,markerfacecolor=c,markeredgecolor='#777',markeredgewidth=.4,label=DISPLAY.get(n,n)) for n,c in CELL_TYPES.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False,fontsize=10.5,columnspacing=2.2,title='Cell type')
    ax=box(fig,.69,.20,.29,.70);types=list(CELL_TYPES);matrix=routes.pivot(index='sender_type',columns='receiver_type',values='percent_all_edges').reindex(index=types,columns=types).fillna(0)
    total=matrix.sum(axis=1).to_numpy();xmax=max(10,np.ceil(total.max()/10)*10+10)
    ax.barh(range(6),[xmax]*6,color='#f3f3f3',height=.68);left=np.zeros(6)
    for n,c in CELL_TYPES.items():
        vals=matrix[n].to_numpy();ax.barh(range(6),vals,left=left,color=c,height=.68)
        for i,v in enumerate(vals):
            if v>=8:ax.text(left[i]+v/2,i,f'{v:.1f}',ha='center',va='center',fontsize=10)
        left+=vals
    for i,v in enumerate(total):ax.text(v+.8,i,f'{v:.1f}%',va='center',fontsize=10)
    ax.set(yticks=range(6),yticklabels=[DISPLAY.get(t,t) for t in types],xlim=(0,xmax),xlabel='% of all 302 edges')
    ax.invert_yaxis();ax.tick_params(labelsize=10.5);ax.text(.5,1.04,'Sender → receiver',transform=ax.transAxes,ha='center',fontsize=10.5)
    return fig

def catalogue(which):
    hits=read('catalog_top20_hits_PSAPP.csv');x=np.arange(4);fig,ax=plt.subplots(figsize=(5.52,2.90))
    if which=='e':
        d=hits[hits.catalog.eq('OmniPath')].set_index('method').loc[METHODS];values=d.lift_over_pool
        bars=ax.bar(x,values,.34,color='#91b5cc',edgecolor='#555',lw=.6);ax.bar_label(bars,labels=[f'{v:.2f}×' for v in values],padding=3,fontsize=10)
        ax.set_ylabel('Enrichment over\ncandidate pool',fontsize=10.5);ax.set_ylim(0,max(values)*1.25)
        ax.set_xticks(x,[f'{label}\nPool: {int(n):,}' for label,n in zip(LABELS,d.pool_n)])
        fig.subplots_adjust(left=.16,right=.98,bottom=.26,top=.83)
    else:
        for j,(cat,c) in enumerate([('OmniPath','#fbb4ae'),('CellChat','#8dd3c7')]):
            d=hits[hits.catalog.eq(cat)].set_index('method').loc[METHODS]
            bars=ax.bar(x+(2*j-1)*.20,d.enrichment_over_null,.40,color=c,edgecolor='#555',lw=.6,label=cat)
            ax.bar_label(bars,labels=[f'{v:.1f}×' for v in d.enrichment_over_null],padding=3,fontsize=9)
        ax.set_ylabel('Enrichment over\nL×R null',fontsize=10.5);ax.set_ylim(0,hits.enrichment_over_null.max()*1.22);ax.set_xticks(x,LABELS)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.27),ncol=2,frameon=False,fontsize=10)
        fig.subplots_adjust(left=.16,right=.98,bottom=.18,top=.79)
    ax.axhline(1,color='#888',lw=.9,ls=(0,(4,3)));ax.tick_params(labelsize=9.5);ax.set_xlim(-.55,3.55)
    return fig

def panel_e():return catalogue('e')
def panel_f():return catalogue('f')
PANELS=dict(a=panel_a,b=panel_b,c=panel_c,d=panel_d,e=panel_e,f=panel_f)
def complete(figures):
    return combine(figures,dict(a=(.045,.035,.94,.27),b=(.045,.34,.94,.25),c=(.045,.66,.16,.33),d=(.225,.66,.50,.33),e=(.735,.66,.25,.16),f=(.735,.845,.25,.145)),(20,15.05))
