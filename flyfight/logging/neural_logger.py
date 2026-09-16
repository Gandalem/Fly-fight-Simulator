def summarize(brain,duration):
    pops=brain.graph.populations
    return {name:dict(spike_count=int(brain.total_spikes[ix].sum()),
                      mean_firing_rate=float(brain.total_spikes[ix].mean()/duration) if len(ix) else None,
                      neurons=len(ix)) for name,ix in pops.items()}
