"""Shared spatial and receptor-to-target drawing geometry."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle,PathPatch
from matplotlib.path import Path as MplPath
NETWORK_STAGES=['#7faac9','#b6aed1','#e0bd95']
NETWORK_HIGHLIGHT='#c4aa85'
LINK_ALPHA,HIGHLIGHT_ALPHA=.55,.65
SPATIAL_PALETTE=dict(blue='#7e9faf',green='#ccebc5',coral='#e0bd95',purple='#decbe4',orange='#fed9a6',yellow='#ffffcc',gray='#dce1e3')
CELL_TYPES={'Astrocyte':('#fbb4ae','o'),'Microglial Cell':('#b3cde3','s'),'Neuron':('#ccebc5','^'),'Oligodendrocyte':('#decbe4','P'),'Pericyte':('#fed9a6','D'),'Schwann Cell':('#ffffcc','>')}
MARKER_CMAP=LinearSegmentedColormap.from_list('marker',['#b3cde3','#eeeeee','#fbb4ae'])
def spatial_axes(ax, control):
    """Identical bounds, equal aspect and downward image y for all three maps."""
    xy = control[['x', 'y']].to_numpy(dtype=float)
    if len(xy) == 0 or not np.isfinite(xy).all():
        raise ValueError('Spatial map requires finite control coordinates')
    low, high = xy.min(axis=0), xy.max(axis=0)
    pad = np.maximum(high-low, 1) * .025
    ax.set(xlim=(low[0]-pad[0], high[0]+pad[0]),
           ylim=(high[1]+pad[1], low[1]-pad[1]), aspect='equal')
    ax.axis('off')

def cell_type_map(control):
    missing = set(control.cell_type) - set(CELL_TYPES)
    if missing or control.cell_type.isna().any():
        raise ValueError(f'Unmapped cell types: {missing}')
    fig = plt.figure(figsize=(6.0, 3.85))
    ax = fig.add_axes([.01, .02, .70, .96])
    for name, (color, marker) in CELL_TYPES.items():
        cells = control[control.cell_type.eq(name)]
        ax.scatter(cells.x, cells.y, c=color, marker=marker, s=5,
                   linewidths=0, label=name, rasterized=True)
    spatial_axes(ax, control)
    ax.legend(loc='center left', bbox_to_anchor=(1.015, .5), ncol=1,
              frameon=False, fontsize=11, markerscale=2.2, handletextpad=.35,
              borderaxespad=0)
    return fig
def sankey(ax,rt,tg,label_fontsize=7,label_spacing=None):
    """Muted blue/teal/coral nodes and score-proportional filled ribbons."""
    layers=[sorted(rt.receptor.unique()),sorted(set(rt.tf)&set(tg.tf)),sorted(tg.target.unique())]
    links=[]
    for frame,l,r,col in [(rt,'receptor','tf',0),(tg,'tf','target',1)]:
        maxima=frame.groupby(l).value.max()
        for row in frame.itertuples():
            if getattr(row,r) not in layers[col+1] or getattr(row,l) not in layers[col]:continue
            links.append((col,getattr(row,l),getattr(row,r),row.value,row.value==maxima[getattr(row,l)]))
    weights={};coords={};padding=.009
    for col,names in enumerate(layers):
        vals={n:max(sum(v for c,a,b,v,top in links if c==col and a==n),sum(v for c,a,b,v,top in links if c+1==col and b==n),1e-9) for n in names}
        weights.update({(col,n):v for n,v in vals.items()})
    scale=min((.90-padding*(len(ns)-1))/sum(weights[col,n] for n in ns) for col,ns in enumerate(layers))
    for col,names in enumerate(layers):
        total=sum(weights[col,n]*scale for n in names)+padding*(len(names)-1);y=(1+total)/2
        for n in names:h=weights[col,n]*scale;coords[col,n]=(col,y-h,h);y-=h+padding
    usedout={k:0. for k in coords};usedin={k:0. for k in coords}
    for c,a,b,value,top in links:
        x,y,h=coords[c,a];xx,yy,hh=coords[c+1,b];w=value*scale
        yl=y+usedout[c,a];yr=yy+usedin[c+1,b];usedout[c,a]+=w;usedin[c+1,b]+=w
        path=MplPath([(x+.025,yl),(x+.4,yl),(xx-.4,yr),(xx-.025,yr),(xx-.025,yr+w),(xx-.4,yr+w),(x+.4,yl+w),(x+.025,yl+w),(x+.025,yl)],
            [MplPath.MOVETO,*([MplPath.CURVE4]*3),MplPath.LINETO,*([MplPath.CURVE4]*3),MplPath.CLOSEPOLY])
        ax.add_patch(PathPatch(path,facecolor=NETWORK_HIGHLIGHT if top else NETWORK_STAGES[c],alpha=HIGHLIGHT_ALPHA if top else LINK_ALPHA,edgecolor='none'))
    label_y={key:y+h/2 for key,(x,y,h) in coords.items()}
    if label_spacing is not None:
        # Separate labels vertically while retaining the weighted node geometry.
        for c,names in enumerate(layers):
            keys=sorted([(c,n) for n in names],key=lambda key:label_y[key])
            positions=np.array([label_y[key] for key in keys])
            gap=min(label_spacing,.96/max(1,len(keys)-1))
            for i in range(1,len(positions)):
                positions[i]=max(positions[i],positions[i-1]+gap)
            positions-=max(0,positions[-1]-.98)
            for i in range(len(positions)-2,-1,-1):
                positions[i]=min(positions[i],positions[i+1]-gap)
            positions+=max(0,.02-positions[0])
            label_y.update(zip(keys,positions))
    for (c,n),(x,y,h) in coords.items():
        ax.add_patch(Rectangle((x-.025,y),.05,max(h,.001),color=NETWORK_STAGES[c],lw=.2))
        text_x=x+(.055 if c<2 else -.055)
        text_y=label_y[c,n]
        if label_spacing is not None and abs(text_y-(y+h/2))>1e-6:
            ax.plot([x+(.025 if c<2 else -.025),text_x],[y+h/2,text_y],color='#777777',lw=.5,zorder=4)
        ax.text(text_x,text_y,n,ha='left' if c<2 else 'right',va='center',
                fontsize=min(label_fontsize,270/max(1,len(layers[c]))),color='#222222',zorder=5,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.3) if label_spacing is not None else None)
    for x,t in enumerate(['Receptors','Transcription Factors','Target Genes']):ax.text(x,1.045,t,ha='center',fontsize=12 if label_spacing is not None else 8,weight='bold',color='#2e4668')
    ax.set(xlim=(-.1,2.1),ylim=(-.02,1.08));ax.axis('off')

def compact_sankey(rt,tg):
    common=set(rt.tf)&set(tg.tf)
    tfs=rt[rt.tf.isin(common)].groupby('tf').value.max().nlargest(6).index
    r=rt[rt.tf.isin(tfs)];t=tg[tg.tf.isin(tfs)]
    receptors=r.groupby('receptor').value.sum().nlargest(8).index
    targets=t.groupby('target').value.sum().nlargest(20).index
    r=r[r.receptor.isin(receptors)];t=t[t.target.isin(targets)]
    tfs=set(r.tf)&set(t.tf)
    return r[r.tf.isin(tfs)],t[t.tf.isin(tfs)]
