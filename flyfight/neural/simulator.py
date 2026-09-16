from pathlib import Path
import numpy as np
from .lif import lif_step
from .plasticity import select_plastic, update
from ..agents.brain import FlyBrainState
from ..runtime import Monitor

def smoke(graph,config,steps=100,agents=2,stimulate=None):
    if steps<1 or agents<1: raise ValueError('Positive steps and agents required')
    monitor=Monitor(config)
    plastic=select_plastic(graph,config)
    brains=[FlyBrainState(graph,plastic,config['seed']+i,config) for i in range(agents)]
    if stimulate:
        ids=graph.metadata.neuron_id
        missing=set(stimulate)-set(ids)
        if missing: raise ValueError(f'Unknown neuron IDs: {missing}')
        selected=np.flatnonzero(ids.isin(stimulate))
    else:
        selected=graph.populations['sensory'][:128]
        if not len(selected): selected=np.arange(min(16,graph.n))
    drive=np.zeros(graph.n,np.float32); drive[selected]=250
    # Warm JIT without mutating experimental brain state.
    from .lif import propagate
    propagate(graph.indptr,graph.indices,graph.weights,np.empty(0,np.int64),np.zeros(graph.n,np.float32))
    events=[]
    for t in range(steps):
        for i,b in enumerate(brains):
            spikes=lif_step(b,drive,config)
            update(b,0.,config,False)
            if i==0 and t<1000:
                idx=np.flatnonzero(spikes)[:10000]
                events.extend((t,int(n)) for n in idx)
    Path('results').mkdir(exist_ok=True)
    np.savez_compressed('results/smoke_spikes.npz',events=np.asarray(events,dtype=np.int32).reshape(-1,2),dt=config['neural']['dt'],neuron_ids=graph.metadata.neuron_id.to_numpy())
    assert all(b.graph is graph for b in brains)
    assert all(np.isfinite(b.voltage).all() for b in brains)
    duration=steps*config['neural']['dt']
    result=dict(graph=graph.stats(agents,len(plastic.edges)),agents=agents,steps=steps,
                shared_topology=True,plastic_synapses=len(plastic.edges),
                state_mib_per_agent=brains[0].nbytes/2**20,
                spike_counts=[int(b.total_spikes.sum()) for b in brains],
                stimulated_neuron_ids=graph.metadata.neuron_id.iloc[selected].tolist(),
                population_rates_hz={name:float(brains[0].total_spikes[ix].mean()/duration) if len(ix) else None for name,ix in graph.populations.items()},
                measured=monitor.sample(steps*agents))
    (Path('results')/'smoke.json').write_text(__import__('json').dumps(result,indent=2))
    return result
