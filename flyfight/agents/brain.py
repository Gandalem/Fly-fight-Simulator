import numpy as np

class FlyBrainState:
    def __init__(self, graph, plastic, seed, config):
        self.graph,self.plastic=graph,plastic
        self.rest_voltage=config['neural']['reset_voltage']
        self.rng=np.random.default_rng(seed)
        self.voltage=np.full(graph.n,config['neural']['reset_voltage'],np.float32)
        self.current=np.zeros(graph.n,np.float32)
        self.refractory=np.zeros(graph.n,np.float32)
        self.spikes=np.zeros(graph.n,bool)
        self.pre_trace=np.zeros(graph.n,np.float32)
        self.post_trace=np.zeros(graph.n,np.float32)
        self.total_spikes=np.zeros(graph.n,np.uint64)
        self.delta=np.zeros(len(plastic.edges),np.float32)
        self.eligibility=np.zeros(len(plastic.edges),np.float32)

    def reset(self, mode='B', decay=0.8):
        if mode not in ('A','B','C'): raise ValueError(mode)
        # B preserves ALL neural state, including eligibility and membrane state.
        if mode=='A':
            for name in ['voltage','current','refractory','spikes','pre_trace','post_trace','delta','eligibility']:
                getattr(self,name).fill(0)
            self.voltage.fill(self.rest_voltage)
        elif mode=='C':
            for name in ['delta','eligibility','pre_trace','post_trace']:
                getattr(self,name)[:] *= decay
        self.total_spikes.fill(0)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x,np.ndarray))
