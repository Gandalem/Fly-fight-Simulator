from dataclasses import dataclass
import numpy as np
import pandas as pd
from .metadata import populations

@dataclass
class SharedConnectome:
    """Outgoing CSR: row=pre, indices=post. Immutable shared arrays."""
    indptr: np.ndarray
    indices: np.ndarray
    weights: np.ndarray
    metadata: pd.DataFrame
    provenance: dict

    def __post_init__(self):
        n = len(self.metadata)
        if len(self.indptr) != n+1 or self.indptr[-1] != len(self.indices) or len(self.indices) != len(self.weights):
            raise ValueError('Invalid CSR shapes')
        if np.any(np.diff(self.indptr) < 0) or (len(self.indices) and (self.indices.min() < 0 or self.indices.max() >= n)):
            raise ValueError('Invalid CSR indices')
        if not np.isfinite(self.weights).all():
            raise ValueError('Nonfinite connection weights')
        if self.metadata.neuron_id.duplicated().any():
            raise ValueError('Duplicate neuron IDs')
        for x in [self.indptr, self.indices, self.weights]:
            x.flags.writeable = False
        self.populations = populations(self.metadata)

    @property
    def n(self): return len(self.metadata)

    @property
    def m(self): return len(self.indices)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in [self.indptr, self.indices, self.weights])

    def induced_subset(self, indices):
        from scipy.sparse import csr_matrix
        ix=np.unique(np.asarray(indices,dtype=np.int64))
        mat=csr_matrix((self.weights,self.indices,self.indptr),shape=(self.n,self.n),copy=False)
        sub=mat[ix][:,ix].tocsr()
        return SharedConnectome(sub.indptr,sub.indices,sub.data,self.metadata.iloc[ix].reset_index(drop=True),
                                dict(self.provenance,subset_of_neurons=self.n,subset_size=len(ix)))

    def stats(self, agents=2, plastic=20000):
        return dict(neuron_count=self.n, connection_count=self.m,
                    neuron_type_count=int(self.metadata.type.nunique()),
                    neuron_types=self.metadata.type.value_counts().head(20).to_dict(),
                    populations={k: len(v) for k,v in self.populations.items()},
                    neurotransmitters=self.metadata.neurotransmitter.value_counts().to_dict(),
                    topology_mib=self.nbytes/2**20,
                    estimated_state_mib=agents*(self.n*29+plastic*16)/2**20,
                    estimated_cpu_vram_mib=0, provenance=self.provenance)
