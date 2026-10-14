"""Manuscript Figure 2: matched benchmark, spatial maps and positive densities."""
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb,to_rgba
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
import seaborn as sns
from .common import read,combine,DATASETS,DATASET_LABELS,METHODS,LABELS,COLORS

OFFSETS=np.array([-.20,.12,-.11,.21,.04,-.04,-.26])
def panel_a():
    data=read('benchmark.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.6,4.8))
    fig.subplots_adjust(left=.14,right=.99,bottom=.16,top=.95,wspace=.32)
    for ax,metric,label in zip(axes,['expression_r','expression_rmse'],['Pearson r of feature means →','RMSE of feature means ←']):
        means=data.groupby('method')[metric].mean().sort_values(ascending=metric=='expression_rmse')
        for i,m in enumerate(means.index):
            vals=data[data.method.eq(m)].set_index('dataset').loc[DATASETS,metric].to_numpy()
            ax.barh(i,vals.mean(),height=.52,color=to_rgba(COLORS[m],.75 if m=='CellOT' else 1))
            ax.errorbar(vals.mean(),i,xerr=vals.std(ddof=1),fmt='none',ecolor='#555',capsize=3,lw=1)
            ax.scatter(vals,i+OFFSETS*.6,s=13,color='#858585',alpha=.8,zorder=3)
        for v in [means.min(),means.max()]:ax.axvline(v,color='#888',ls='--',lw=1)
        ax.set(yticks=range(5),yticklabels=[LABELS[m] for m in means.index],ylim=(4.55,-.55),xlabel=label)
        ax.set_xlim((.5,1.01) if metric=='expression_r' else (0,.58))
        ax.tick_params(axis='y',length=0,labelsize=12)
    return fig

def panel_b():
    df=read('benchmark.csv');fig,axes=plt.subplots(1,3,figsize=(8,4.8),sharey=True)
    fig.subplots_adjust(left=.125,right=.985,bottom=.17,top=.86,wspace=.22)
    for k,(ax,metric,title) in enumerate(zip(axes,['response_auc','Corr','cosine_mean'],['Response-direction AUC','Spatial Moran r','Population TF cosine'])):
        median=float(df[df.method.eq('spike')][metric].median())
        ax.plot([median,median],[0,4.55],color='#888',ls=(0,(4,3)),lw=1.05)
        for row,m in enumerate(METHODS):
            vals=df[df.method.eq(m)].set_index('dataset').loc[DATASETS,metric].to_numpy()
            face=to_rgba(COLORS[m],.75 if m=='CellOT' else 1)
            ax.axhline(row,color='#eee',lw=.7,zorder=0)
            ax.scatter(vals,row+OFFSETS,s=21,color=face,edgecolors='#888',linewidths=.4,zorder=3)
            ax.errorbar(vals.mean(),row,xerr=vals.std(ddof=1),fmt='o',mfc=face,mec='#333',mew=.9,ms=8,ecolor='#555',elinewidth=1.2,capsize=3,zorder=5)
        ax.set(xlim=(-.035,1.07),ylim=(4.55,-.57),xticks=[0,.5,1],xticklabels=['0','0.5','1.0'],yticks=range(5))
        ax.set_title(title,fontsize=11.6,pad=15);ax.tick_params(axis='y',length=0,pad=10,labelsize=14);ax.tick_params(axis='x',labelsize=12)
        ax.spines['left'].set_visible(False);ax.spines['bottom'].set_bounds(0,1)
        if k==0:ax.set_yticklabels([LABELS[m] for m in METHODS])
    fig.text(.555,.055,'Score →',ha='center',fontsize=12)
    return fig

def panel_c():
    values=read('benchmark_spatial.csv');methods=['Source','Spike','State','scGPT','scGen','CellOT']
    fig,axes=plt.subplots(1,6,figsize=(25,4.8),layout='constrained');high=values.expression.quantile(.99)
    for ax,m in zip(axes,methods):
        d=values[values.method.eq(m)]
        im=ax.scatter(d.x,d.y,c=d.expression,s=28,cmap=sns.color_palette('vlag',as_cmap=True),vmin=0,vmax=high,linewidths=0,rasterized=True)
        ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');ax.text(.5,1.02,m,transform=ax.transAxes,ha='center',fontsize=20)
    fig.colorbar(im,ax=axes,shrink=.7,fraction=.025,label='log1p CPM10k',extend='max')
    return fig

def curves(frame,order):
    kernels={}
    for m in order:
        x=frame.loc[frame.Source.eq(m),'Expression'].to_numpy(float);x=x[x>0]
        kde=gaussian_kde(x,bw_method=lambda k:1.5*k.scotts_factor())
        kernels[m]=(kde,x.max()+4*np.sqrt(kde.covariance[0,0]))
    grid=np.linspace(0,max(v[1] for v in kernels.values()),4000)
    return {m:(grid,(k(grid)+k(-grid))/np.max(k(grid)+k(-grid))) for m,(k,_) in kernels.items()}

def panel_d():
    data=read('benchmark_distributions.csv');order=['Control','Treated','spike','state','scGPT','scGen','CellOT']
    palette={'Control':'gray','Treated':'blue'}
    for m in METHODS:
        h,l,s=colorsys.rgb_to_hls(*to_rgb(COLORS[m]));palette[m]=colorsys.hls_to_rgb(h,l-.14,s) if m!='CellOT' else '#f2c230'
    fig,axes=plt.subplots(1,2,figsize=(11,3.4),sharey=True);fig.subplots_adjust(left=.08,right=.78,bottom=.18,top=.88,wspace=.16)
    for ax,gene in zip(axes,['Sgk1','Ttr']):
        for m,(x,y) in curves(data[data.Gene.eq(gene)],order).items():ax.plot(x,y,color=palette[m],lw=2,label=LABELS.get(m,m),alpha=.75 if m=='CellOT' else .95)
        ax.set(xlim=(-.03*x[-1],1.03*x[-1]),ylim=(-.04,1.04),yticks=[0,.25,.5,.75,1],xlabel='Expression (log1p CPM10k)')
        ax.set_title(gene,fontsize=18,fontstyle='italic',pad=10)
    axes[0].set_ylabel('Relative density');fig.legend(*axes[0].get_legend_handles_labels(),loc='center left',bbox_to_anchor=(.80,.5),frameon=False)
    return fig

def panel_e():
    table=read('benchmark.csv').pivot(index='dataset',columns='method',values='w1').loc[DATASETS,METHODS]
    fig,axes=plt.subplots(1,5,figsize=(6.4,3.94),sharey=True);fig.subplots_adjust(left=.20,right=.975,bottom=.20,top=.81,wspace=.19)
    for i,(ax,m) in enumerate(zip(axes,METHODS)):
        c=to_rgba(COLORS[m],.75 if m=='CellOT' else 1);ax.set_facecolor('#fafafa')
        ax.barh(range(7),table[m],height=.58,color=c,edgecolor='#777',linewidth=.35,zorder=3)
        ax.add_patch(Rectangle((0,1.045),1,.12,transform=ax.transAxes,facecolor=c,edgecolor='none',clip_on=False))
        ax.text(.5,1.102,LABELS[m],transform=ax.transAxes,ha='center',va='center',fontsize=11.5)
        ax.set(xlim=(0,.30),ylim=(6.55,-.55),yticks=range(7),xticks=[0,.3],xticklabels=['0','0.3'])
        ax.tick_params(axis='x',labelsize=10);ax.tick_params(axis='y',labelsize=10.5,length=0,pad=8)
        ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
        if i==0:ax.set_yticklabels([DATASET_LABELS[d] for d in DATASETS])
        ax.spines['left'].set_visible(False);ax.grid(axis='y',color='white');ax.set_axisbelow(True)
    fig.text(.59,.065,'Mean $W_1$ distance ←',ha='center',fontsize=12)
    return fig

PANELS=dict(a=panel_a,b=panel_b,c=panel_c,d=panel_d,e=panel_e)
def complete(figures):
    return combine(figures,dict(a=(.03,.035,.465,.35),b=(.51,.035,.475,.35),c=(.03,.42,.955,.25),d=(.03,.72,.60,.265),e=(.65,.72,.335,.265)),(18,13.72))
