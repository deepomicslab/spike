"""Checked numerical inputs and fresh, isolated figure outputs."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('SPIKE_FIGURE_DATA', ROOT / 'reproduction')).expanduser().resolve()
OUTPUT = Path(os.environ.get('SPIKE_FIGURE_OUTPUT', ROOT / 'outputs/reproduction')).expanduser().resolve()
DATASETS = ['brain_sleep1', 'brain_sleep2', 'brain_sleep3', 'brain_sleep4', 'WT', 'SHH', 'PSAPP']
# Keep the released table/provenance keys stable; use manuscript names for display.
DATASET_LABELS = {
    'brain_sleep1': 'RHY 1', 'brain_sleep2': 'RHY 2',
    'brain_sleep3': 'RHY 3', 'brain_sleep4': 'RHY 4',
    'WT': 'WT', 'SHH': 'SHH', 'PSAPP': 'PSAPP',
}
METHODS = ['spike', 'state', 'scGen', 'scGPT', 'CellOT']
LABELS = dict(spike='Spike', state='State', scGen='scGen', scGPT='scGPT', CellOT='CellOT')
COLORS = dict(spike='#b3cde3', state='#ccebc5', scGen='#decbe4', scGPT='#fed9a6', CellOT='#f2c230')
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42}
plt.rcParams.update(STYLE)

def display_table(frame):
    """Return a presentation-only copy without changing numerical input keys."""
    result = frame.copy()
    for column in ('dataset', 'Dataset'):
        if column in result.columns:
            result[column] = result[column].replace(DATASET_LABELS)
    return result

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(name):
    """Reject changed inputs before drawing, rather than checking old images."""
    manifest = json.loads((DATA / 'manifest.json').read_text())
    path = DATA / 'tables' / name
    if sha(path) != manifest['tables'][name]:
        raise ValueError(f'Changed reproduction table: {path}')
    return pd.read_csv(path)

def audit():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name in manifest['tables']:
        if sha(DATA / 'tables' / name) != manifest['tables'][name]:
            raise ValueError(f'Changed reproduction table: {name}')
    record = json.loads((ROOT / 'provenance/current_spike_model.json').read_text())
    if sha(ROOT / record['packaged_module']) != record['packaged_sha256']:
        raise ValueError('Published model source hash differs')
    print(f"Input audit: PASS ({len(manifest['tables'])} numerical tables)")
    print('Figure protocol:', manifest['protocol'])
    return manifest

def save(fig, number, panel):
    directory = OUTPUT / f'figure{number}'
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(directory / f'{panel}.{ext}', dpi=180, facecolor='white')
    return fig

def box(fig, x, y, w, h):
    return fig.add_axes([x, 1-y-h, w, h])

def combine(figures, layout, size, labels=True):
    """Assemble *newly drawn canvases* in memory. No reference artwork is read."""
    page = plt.figure(figsize=size, facecolor='white')
    for key, (x,y,w,h) in layout.items():
        fig = figures[key]
        # Inline backends can detach the canvas when a displayed figure closes.
        canvas = FigureCanvasAgg(fig)
        previous_dpi = fig.dpi
        fig.set_dpi(180)
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba()).copy()
        fig.set_dpi(previous_dpi)
        ax = box(page, x, y, w, h)
        ax.imshow(rgba, aspect='auto', interpolation='lanczos')
        ax.axis('off')
        if labels:
            page.text(max(.002,x-.012),min(.97,1-y+.008),key.upper()+'.',fontsize=15,weight='bold',va='bottom')
    return page

def finish(number, panels):
    directory = OUTPUT / f'figure{number}'
    expected = [directory / f'{key}.{ext}' for key in [*panels, 'complete'] for ext in ('png','pdf','svg')]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing: raise FileNotFoundError('Run every panel first: '+', '.join(missing))
    report = {'status':'PASS', 'figure':number, 'panels':list(panels),
              'input_manifest_sha256':sha(DATA/'manifest.json'),
              'outputs':{p.name:sha(p) for p in expected},
              'plotting_code':{p.name:sha(p) for p in sorted((ROOT/'reproduction').glob('*.py'))}}
    (directory/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Figure {number}: PASS — {len(panels)} panels redrawn from numerical tables')
    print('Outputs:',directory)
    return report
