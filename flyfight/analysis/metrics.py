import pandas as pd
import numpy as np
from ..logging.episode_logger import read

def frame(path):
    rows=[]; previous={}; opponents={}
    for ep in read(path):
        for a in ep['agents']:
            i=a['fly_id']; other=tuple(x['fly_id'] for x in ep['agents'] if x['fly_id']!=i)
            opponents.setdefault(i,set()).update(other)
            outcome='draw' if ep['winner'] is None else 'win' if ep['winner']==i else 'loss'
            r={k:v for k,v in a.items() if not isinstance(v,(dict,list))}
            r.update(episode=ep['episode_id'],duration=ep['duration'],win=float(outcome=='win'),
                     outcome=outcome,previous_outcome=previous.get(i,'none'),
                     opponent_diversity=len(opponents[i]),plasticity_on=ep['plasticity_on'],
                     attack_rate=a['attack_attempts']/ep['duration'],retreat_rate=a['retreat_count']/ep['duration'],
                     mean_abs_delta=a['plasticity']['mean_abs_delta'],modified_synapses=a['plasticity']['modified_synapses'])
            rows.append(r); previous[i]=outcome
    return pd.DataFrame(rows)

def pca(features):
    x=np.asarray(features,dtype=float)
    x=(x-x.mean(0))/np.maximum(x.std(0),1e-9)
    u,s,v=np.linalg.svd(x,full_matrices=False)
    if len(s): s[s<max(1e-12,s[0]*1e-10)]=0
    result=u[:,:2]*s[:2]
    if result.shape[1]<2: result=np.pad(result,((0,0),(0,2-result.shape[1])))
    return result
