from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PlasticSubset:
    edges: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    base: np.ndarray

def select_plastic(graph, config):
    c=config['learning']
    candidate=graph.metadata['class'].isin(c['candidate_classes']).to_numpy()
    descending=np.zeros(graph.n,dtype=bool)
    if c['include_descending_targets']: descending[graph.populations['descending']]=True
    # Bounded candidate collection and deterministic top-strength selection.
    chosen=np.empty(0,dtype=np.int64)
    k=c['max_synapses']
    for lo in range(0,graph.n,1024):
        hi=min(graph.n,lo+1024)
        start,end=graph.indptr[lo],graph.indptr[hi]
        pres=np.repeat(np.arange(lo,hi),np.diff(graph.indptr[lo:hi+1]))
        posts=graph.indices[start:end]
        valid=((candidate[pres] & candidate[posts]) | descending[posts]) & (graph.weights[start:end]!=0)
        edges=np.flatnonzero(valid)+start
        chosen=np.concatenate([chosen,edges])
        if len(chosen)>k:
            # Stable tie-break by edge index; avoids arbitrary IDs or topology additions.
            order=np.lexsort((chosen,-np.abs(graph.weights[chosen])))[:k]
            chosen=chosen[order]
    chosen.sort()
    pre=(np.searchsorted(graph.indptr,chosen,side='right')-1).astype(np.int32)
    arrays=[chosen,pre,graph.indices[chosen].copy(),graph.weights[chosen].copy()]
    for x in arrays: x.flags.writeable=False
    return PlasticSubset(*arrays)

def update(brain, modulation, config, enabled=True):
    c=config['learning']; dt=config['neural']['dt']
    p=brain.plastic
    brain.eligibility *= np.float32(np.exp(-dt/c['tau_eligibility']))
    # Use traces BEFORE adding this step's spikes: genuinely causal STDP.
    brain.eligibility += c['a_plus']*brain.pre_trace[p.pre]*brain.spikes[p.post] - c['a_minus']*brain.post_trace[p.post]*brain.spikes[p.pre]
    decay=np.float32(np.exp(-dt/c['tau_trace']))
    brain.pre_trace *= decay; brain.post_trace *= decay
    brain.pre_trace += brain.spikes; brain.post_trace += brain.spikes
    np.clip(brain.eligibility,-10,10,out=brain.eligibility)
    if enabled:
        brain.delta += np.float32(c['learning_rate']*modulation*dt)*brain.eligibility
        bound=np.abs(p.base)*c['max_relative_change']
        np.clip(brain.delta,-bound,bound,out=brain.delta)

def statistics(brain):
    d=brain.delta
    order=np.argsort(np.abs(d))[-10:][::-1]
    return dict(modified_synapses=int(np.count_nonzero(d)),
                mean_abs_delta=float(np.mean(np.abs(d))) if len(d) else 0,
                max_abs_delta=float(np.max(np.abs(d))) if len(d) else 0,
                largest_changes=[dict(edge=int(brain.plastic.edges[i]),pre=int(brain.graph.metadata.neuron_id.iloc[brain.plastic.pre[i]]),
                                      post=int(brain.graph.metadata.neuron_id.iloc[brain.plastic.post[i]]),delta=float(d[i])) for i in order])
