import numpy as np
from .motor_learning import ACTION_NAMES

class MotorAdapter:
    """Read only CNS activity. No food-seeking or opponent policy outside brain.

    The default readout learns continuous walking, front-joint and adhesion
    outputs. The old fixed decoder is retained only for historical controls.
    Neither mapping is an identified biological descending command pathway.
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
        half=config['motor_learning']['feature_groups']//2
        self.learned_groups=list(np.array_split(left,half))+list(np.array_split(right,half))

    def decode(self,spike_counts,duration,brain=None):
        c=self.config['adapter']
        if self.config['motor_learning']['enabled']:
            if brain is None: raise ValueError('Learned decoder requires independent brain state')
            rates=np.array([float(spike_counts[ix].mean()/duration) if len(ix) else 0. for ix in self.learned_groups])
            rates=np.tanh(rates/c['firing_rate_scale'])
            action=brain.motor.decide(rates,duration,
                explore=self.config['motor_learning']['exploration'] and not self.config.get('evaluation'),
                learn=self.learning_enabled)
            return dict(legs=(c['motor_gain']*action[:2]).tolist(),
                        front_joints=action[2:8].reshape(2,3).tolist(),
                        front_adhesion=((action[8:]+1)/2).tolist(),
                        motor_effort=float(np.mean(np.abs(action[2:8]))),
                        lunge=0.,activity=dict(descending_mean=float(rates.mean())))
        rates={k:float(spike_counts[ix].mean()/duration) if len(ix) else 0. for k,ix in self.groups.items()}
        rates={k:np.tanh(v/c['firing_rate_scale']) for k,v in rates.items()}
        reverse=max(0,rates['retreat']-.5)*1.5
        left=c['motor_gain']*(rates['left']-reverse)
        right=c['motor_gain']*(rates['right']-reverse)
        lunge=max(0,rates['lunge']-.65)/.35
        legs=np.clip(np.array([left,right])*(1+.4*lunge),-1.5,1.5)
        legs[np.abs(legs)<c['motor_deadzone']]=0
        return dict(legs=legs.tolist(),lunge=float(lunge),activity=rates)

    @property
    def learning_enabled(self):
        return (self.config['motor_learning']['enabled'] and self.config['training']['plasticity']
                and self.config['motor_learning']['plasticity'] and not self.config.get('evaluation'))

    def feedback(self,brain,reward,dt):
        if self.config['motor_learning']['enabled']: brain.motor.observe(reward,dt)

    def finish(self,brain):
        if self.config['motor_learning']['enabled']: brain.motor.finish(learn=self.learning_enabled)

    def describe(self):
        if self.config['motor_learning']['enabled']:
            return dict(kind='learned_engineering_readout',actions=list(ACTION_NAMES),
                        feature_neuron_ids=[self.graph.metadata.neuron_id.iloc[v].tolist() for v in self.learned_groups],
                        warning='Readout weights are not MaleCNS synapses; no attack policy is supplied.')
        return {k:self.graph.metadata.neuron_id.iloc[v].tolist() for k,v in self.groups.items()}
