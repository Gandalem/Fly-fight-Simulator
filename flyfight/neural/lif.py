import numpy as np
from numba import njit

@njit(cache=True, nogil=True)
def propagate(indptr, indices, weights, spike_indices, out):
    """Only visit outgoing edges of active neurons; O(active edges), O(N) buffer."""
    for pre in spike_indices:
        for e in range(indptr[pre],indptr[pre+1]):
            out[indices[e]] += weights[e]

def lif_step(brain, drive, config):
    if not config['runtime'].get('fast_neural',True):
        return lif_step_reference(brain,drive,config)
    n=config['neural']; p=brain.plastic
    if not hasattr(brain,'_noise_buffer'):
        brain._noise_buffer=np.empty(brain.graph.n,np.float32)
    if n['noise_std']:
        brain.rng.standard_normal(brain.graph.n,dtype=np.float32,out=brain._noise_buffer)
    _lif_kernel(brain.graph.indptr,brain.graph.indices,brain.graph.weights,p.pre,p.post,brain.delta,
                brain.voltage,brain.current,brain.refractory,brain.spikes,brain.total_spikes,
                np.asarray(drive,dtype=np.float32),brain._noise_buffer,bool(n['noise_std']),
                np.float32(n['dt']),np.float32(np.exp(-n['dt']/n['tau_synapse'])),
                np.float32(np.exp(-n['dt']/n['tau_membrane'])),np.float32(n['reset_voltage']),
                np.float32(n['threshold']),np.float32(n['refractory']),np.float32(n['noise_std']*np.sqrt(n['dt'])))
    return brain.spikes


@njit(cache=True,nogil=True)
def _lif_kernel(indptr,indices,weights,plastic_pre,plastic_post,delta,voltage,current,refractory,spikes,total_spikes,
                drive,noise,add_noise,dt,syn_decay,mem_decay,rest,threshold,refractory_time,noise_scale):
    # Preserve float32 rounding at each NumPy operation. No fastmath/reordering.
    for i in range(len(current)): current[i]=np.float32(current[i]*syn_decay)
    for pre in range(len(spikes)):
        if spikes[pre]:
            for edge in range(indptr[pre],indptr[pre+1]):
                post=indices[edge]
                current[post]=np.float32(current[post]+weights[edge])
    for edge in range(len(delta)):
        post=plastic_post[edge]
        value=np.float32(delta[edge]*spikes[plastic_pre[edge]])
        current[post]=np.float32(current[post]+value)
    for i in range(len(voltage)):
        active=refractory[i]<=0
        refractory[i]=max(np.float32(0),np.float32(refractory[i]-dt))
        value=np.float32(rest+np.float32(np.float32(voltage[i]-rest)*mem_decay))
        value=np.float32(value+np.float32(current[i]+np.float32(drive[i]*dt)))
        if add_noise: value=np.float32(value+np.float32(noise[i]*noise_scale))
        if not active: value=rest
        fired=active and value>=threshold
        voltage[i]=rest if fired else value
        if fired: refractory[i]=refractory_time
        spikes[i]=fired
        total_spikes[i]+=fired


def lif_step_reference(brain, drive, config):
    """Original NumPy path, retained for reproducibility and exact comparisons."""
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
