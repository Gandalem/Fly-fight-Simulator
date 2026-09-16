import numpy as np
from flyfight.training.checkpoint import save,restore
from flyfight.agents.fly import Fly
from flyfight.agents.body import BodyCondition
from flyfight.agents.internal_state import InternalState
from flyfight.agents.brain import FlyBrainState
from test_neural import fixture

def test_checkpoint_and_body_restore(tmp_path):
    g,p,c=fixture()
    f=Fly(0,FlyBrainState(g,p,42,c),BodyCondition(c),InternalState())
    rng=np.random.default_rng(3)
    f.brain.delta[:]=.2; f.brain.eligibility[:]=.3; f.body.parts['Head'].current_integrity=0
    f.reset_body(c)
    assert f.body.modifier('Head')==1 and f.brain.delta[0]==np.float32(.2)
    save(tmp_path/'brain.npz',[f],7,rng)
    expected=f.brain.rng.random(); expected_match=rng.random()
    f.brain.delta[:]=0
    assert restore(tmp_path/'brain.npz',[f],rng)==7
    assert f.brain.rng.random()==expected and rng.random()==expected_match
    assert f.brain.delta[0]==np.float32(.2) and f.brain.eligibility[0]==np.float32(.3)
