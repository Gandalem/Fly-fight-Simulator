from pathlib import Path
import pytest
import numpy as np
from flyfight.config import load_config
from flyfight.connectome.loader import load_connectome
from flyfight.neural.plasticity import select_plastic
from flyfight.agents.brain import FlyBrainState
from flyfight.neural.lif import lif_step

@pytest.mark.data
def test_official_full_graph_smoke():
    path=Path('data/processed/malecns')
    if not path.exists(): pytest.skip('Official data not downloaded')
    g=load_connectome(path); c=load_config()
    assert g.n==166700 and g.m==25582938
    p=select_plastic(g,c); b=FlyBrainState(g,p,1,c)
    drive=np.zeros(g.n,np.float32); drive[g.populations['sensory'][:16]]=2000
    for _ in range(5): lif_step(b,drive,c)
    assert b.total_spikes.sum()>16
    assert np.isfinite(b.voltage).all()
    assert len(p.edges)<=c['learning']['max_synapses']

@pytest.mark.data
def test_official_induced_subset():
    path=Path('data/processed/malecns')
    if not path.exists(): pytest.skip('Official data not downloaded')
    graph=load_connectome(path)
    pre=int(graph.populations['descending'][0])
    neighbors=graph.indices[graph.indptr[pre]:graph.indptr[pre+1]]
    g=graph.induced_subset(np.r_[pre,neighbors]); c=load_config()
    assert g.n<=graph.n and g.m>0
    p=select_plastic(g,c); b=FlyBrainState(g,p,1,c)
    drive=np.full(g.n,2000,np.float32)
    for _ in range(5): lif_step(b,drive,c)
    assert b.total_spikes.sum()>0
