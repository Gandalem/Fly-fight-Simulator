import copy
import numpy as np
import pandas as pd
from flyfight.config import load_config
from flyfight.connectome.graph import SharedConnectome
from flyfight.agents.brain import FlyBrainState
from flyfight.neural.plasticity import PlasticSubset, update
from flyfight.neural.lif import lif_step

def fixture():
    g=SharedConnectome(np.array([0,1,1,1]),np.array([1],np.int32),np.array([1.2],np.float32),
        pd.DataFrame(dict(neuron_id=[11,22,33],type=['s','d','m'],superclass=['cb_sensory','descending_neuron','vnc_motor'],
                          neurotransmitter=['acetylcholine']*3,**{'class':['olfactory','MBON','']})),{'dataset':'synthetic-test-only'})
    p=PlasticSubset(np.array([0]),np.array([0]),np.array([1]),np.array([1.2],np.float32))
    c=load_config(); c['neural']['noise_std']=0
    return g,p,c

def test_propagation_and_refractory():
    g,p,c=fixture(); b=FlyBrainState(g,p,0,c)
    assert lif_step(b,np.array([2000,0,0]),c).tolist()==[True,False,False]
    assert lif_step(b,np.zeros(3),c).tolist()==[False,True,False]
    assert not lif_step(b,np.array([2000,0,0]),c)[0]

def test_independent_state_and_reset():
    g,p,c=fixture(); a=FlyBrainState(g,p,0,c); b=FlyBrainState(g,p,1,c)
    a.delta[:]=.1; a.eligibility[:]=.2; a.voltage[:]=.5
    a.reset('B'); assert a.delta[0]==np.float32(.1) and a.voltage[0]==.5
    a.reset('C',.5); assert a.delta[0]==np.float32(.05)
    a.reset('A'); assert not a.delta.any()
    assert a.graph is b.graph and not np.shares_memory(a.voltage,b.voltage)
    assert not g.weights.flags.writeable

def test_stdp_temporal_order_and_control():
    g,p,c=fixture(); a=FlyBrainState(g,p,0,c)
    a.spikes[:]=[1,0,0]; update(a,0,c)
    a.spikes[:]=[0,1,0]; update(a,1,c)
    assert a.eligibility[0]>0 and a.delta[0]>0
    b=FlyBrainState(g,p,0,c)
    b.spikes[:]=[0,1,0]; update(b,0,c)
    b.spikes[:]=[1,0,0]; update(b,1,c)
    assert b.delta[0]<0
    prior=b.delta.copy(); update(b,100,c,False)
    np.testing.assert_array_equal(prior,b.delta)
