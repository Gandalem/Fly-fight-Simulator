"""Steady-state active-graph throughput vs silent-graph throughput."""
import json
import time
import numpy as np
from flyfight.config import load_config
from flyfight.connectome.loader import load_connectome
from flyfight.agents.brain import FlyBrainState
from flyfight.neural.plasticity import select_plastic,update
from flyfight.neural.lif import lif_step,propagate
from flyfight.runtime import Monitor

c=load_config(); g=load_connectome(c['dataset']['processed']); p=select_plastic(g,c)
monitor=Monitor(c)
propagate(g.indptr,g.indices,g.weights,np.array([],np.int64),np.zeros(g.n,np.float32))
results={}
for condition in ['silent','stimulated']:
    brains=[FlyBrainState(g,p,i,c) for i in range(2)]
    drive=np.zeros(g.n,np.float32)
    if condition=='stimulated': drive[g.populations['sensory'][:128]]=120
    for b in brains:
        for _ in range(20): lif_step(b,drive,c)
    start=time.perf_counter()
    for _ in range(100):
        for b in brains:
            lif_step(b,drive,c); update(b,0,c)
    elapsed=time.perf_counter()-start
    results[condition]={'brain_steps_per_sec':200/elapsed,'two_agent_world_steps_per_sec':100/elapsed,
                        'mean_hz':float(brains[0].total_spikes.mean()/.12),'seconds':elapsed}
results['memory']=monitor.sample()
print(json.dumps(results,indent=2))
