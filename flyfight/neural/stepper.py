"""Parallel independent brains with a barrier before every physics interval."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .lif import lif_step
from .plasticity import update


class BrainStepper:
    def __init__(self,config,agents):
        self.config=config
        self.steps=round(config['arena']['control_dt']/config['neural']['dt'])
        self.plasticity=(config['training']['plasticity'] and config['learning']['enabled']
                         and not config.get('evaluation'))
        self.workers=min(agents,config['runtime']['brain_workers'])
        self.executor=None

    def __enter__(self):
        if self.workers>1: self.executor=ThreadPoolExecutor(self.workers,thread_name_prefix='flyfight-brain')
        return self

    def __exit__(self,*exc):
        if self.executor is not None: self.executor.shutdown(wait=True,cancel_futures=True)

    def _advance(self,brain,drive,modulation):
        counts=np.zeros(brain.graph.n,np.uint16)
        for _ in range(self.steps):
            counts+=lif_step(brain,drive,self.config)
            update(brain,modulation,self.config,self.plasticity)
        return counts

    def advance(self,brains,drives,modulation):
        if self.executor is None:
            return [self._advance(*args) for args in zip(brains,drives,modulation)]
        # map preserves agent order. Workers never touch physics, other brains,
        # matchmaking RNG, motor policy RNG, or the read-only shared graph.
        return list(self.executor.map(self._advance,brains,drives,modulation))
