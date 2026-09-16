"""Paired actual-graph GUI/headless regression with exact trajectory outcomes.

Short runs test cadence and scientific-state isolation, not learned strategy.
"""
from pathlib import Path
import json
import numpy as np
from flyfight.config import load_config
from flyfight.connectome.loader import load_connectome
from flyfight.training.trainer import train
from flyfight.logging.episode_logger import read

c=load_config('configs/quick.yaml')
c['arena']['duration']=.3
c['training']['episodes']=1
graph=load_connectome(c['dataset']['processed'])
paths=[]
for gui in [False,True]:
    path=Path(train(graph,c,gui=gui))
    paths.append(path)
headless,gui=map(read,paths)
assert headless[0]['agents']==gui[0]['agents'], 'GUI altered scientific episode outcomes'
with np.load(paths[0]/'checkpoint.npz') as a,np.load(paths[1]/'checkpoint.npz') as b:
    for key in a.files:
        np.testing.assert_array_equal(a[key],b[key],err_msg=f'GUI changed checkpoint {key}')
metrics=json.loads((paths[1]/'viewer_metrics.json').read_text())
complete=json.loads((paths[1]/'complete.json').read_text())
assert not complete['offscreen_renderer_created'], 'GUI unnecessarily created an offscreen renderer'
assert metrics['error'] is None and metrics['frames']>metrics['source_updates']
result=dict(headless=str(paths[0]),gui=str(paths[1]),exact_episode_and_checkpoint_match=True,
            timings=gui[0]['timings'],viewer=metrics,performance=complete['performance'])
Path('results/gui_regression.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
