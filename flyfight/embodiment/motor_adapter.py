import numpy as np

class MotorAdapter:
    """Read only CNS activity. No food-seeking or opponent policy outside brain.

    Left/right DN activity drives CPG amplitudes. Other DN shards enable primitive
    retreat/lunge. These are configurable engineering hypotheses, not identified
    biological descending commands. No weights are trained in this decoder.
    """
    def __init__(self,graph,config):
        self.graph=graph; self.config=config
        dn=graph.populations['descending']
        if len(dn)<8: raise ValueError('At least 8 annotated descending neurons required')
        meta=graph.metadata
        side=meta.somaSide.fillna('').to_numpy() if 'somaSide' in meta else np.full(graph.n,'')
        left=dn[side[dn]=='L']; right=dn[side[dn]=='R']
        if not len(left) or not len(right): left,right=np.array_split(dn,2)
        self.groups=dict(left=left[::2],right=right[::2],retreat=dn[1::4],lunge=dn[3::4])

    def decode(self,spike_counts,duration):
        c=self.config['adapter']
        rates={k:float(spike_counts[ix].mean()/duration) if len(ix) else 0. for k,ix in self.groups.items()}
        rates={k:np.tanh(v/c['firing_rate_scale']) for k,v in rates.items()}
        reverse=max(0,rates['retreat']-.5)*1.5
        left=c['motor_gain']*(rates['left']-reverse)
        right=c['motor_gain']*(rates['right']-reverse)
        lunge=max(0,rates['lunge']-.65)/.35
        legs=np.clip(np.array([left,right])*(1+.4*lunge),-1.5,1.5)
        legs[np.abs(legs)<c['motor_deadzone']]=0
        return dict(legs=legs.tolist(),lunge=float(lunge),activity=rates)

    def describe(self):
        return {k:self.graph.metadata.neuron_id.iloc[v].tolist() for k,v in self.groups.items()}
