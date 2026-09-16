import numpy as np
from numba import njit

@njit(cache=True)
def propagate(indptr, indices, weights, spike_indices, out):
    """Only visit outgoing edges of active neurons; O(active edges), O(N) buffer."""
    for pre in spike_indices:
        for e in range(indptr[pre],indptr[pre+1]):
            out[indices[e]] += weights[e]

def lif_step(brain, drive, config):
    n=config['neural']; dt=n['dt']
    brain.current *= np.float32(np.exp(-dt/n['tau_synapse']))
    propagate(brain.graph.indptr,brain.graph.indices,brain.graph.weights,np.flatnonzero(brain.spikes),brain.current)
    if len(brain.plastic.pre):
        np.add.at(brain.current,brain.plastic.post,brain.delta*brain.spikes[brain.plastic.pre])
    active=brain.refractory <= 0
    brain.refractory[:] = np.maximum(0,brain.refractory-dt)
    decay=np.float32(np.exp(-dt/n['tau_membrane']))
    # Exact exponential membrane leak; white noise scales with sqrt(dt).
    brain.voltage[:] = n['reset_voltage']+(brain.voltage-n['reset_voltage'])*decay
    brain.voltage += brain.current + np.asarray(drive,dtype=np.float32)*dt
    if n['noise_std']:
        brain.voltage += brain.rng.standard_normal(brain.graph.n,dtype=np.float32)*np.float32(n['noise_std']*np.sqrt(dt))
    brain.voltage[~active]=n['reset_voltage']
    brain.spikes[:] = active & (brain.voltage >= n['threshold'])
    brain.voltage[brain.spikes]=n['reset_voltage']
    brain.refractory[brain.spikes]=n['refractory']
    brain.total_spikes += brain.spikes
    return brain.spikes
