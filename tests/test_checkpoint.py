import numpy as np
import pytest
from flyfight.training.checkpoint import save,restore,transfer_weights
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
    f.brain.motor.weights.fill(.4)
    f.brain.motor.decide(np.ones(32),.1)
    f.brain.motor.observe(.1,.1)
    f.reset_body(c)
    assert f.body.modifier('Head')==1 and f.brain.delta[0]==np.float32(.2)
    save(tmp_path/'brain.npz',[f],7,rng)
    expected=f.brain.rng.random(); expected_match=rng.random()
    expected_motor_rng=f.brain.motor.rng.random()
    expected_motor_score=f.brain.motor.score.copy()
    f.brain.delta[:]=0
    f.brain.motor.reset('A')
    assert restore(tmp_path/'brain.npz',[f],rng)==7
    assert f.brain.rng.random()==expected and rng.random()==expected_match
    assert f.brain.delta[0]==np.float32(.2) and f.brain.eligibility[0]==np.float32(.3)
    assert f.brain.motor.rng.random()==expected_motor_rng
    np.testing.assert_array_equal(f.brain.motor.score,expected_motor_score)
    assert np.all(f.brain.motor.weights==np.float32(.4))
    assert f.brain.motor.pending and f.brain.motor.reward==.1


def test_transfer_motor_policy_for_frozen_evaluation(tmp_path):
    g,p,c=fixture()
    trained=Fly(0,FlyBrainState(g,p,42,c),BodyCondition(c),InternalState())
    naive=Fly(1,FlyBrainState(g,p,43,c),BodyCondition(c),InternalState())
    trained.brain.motor.weights.fill(.35)
    trained.brain.motor.value_weights.fill(.2)
    trained.brain.delta.fill(.1)
    trained.brain.voltage.fill(.5)
    path=tmp_path/'learned.npz'; save(path,[trained],9,np.random.default_rng(1))
    transfer_weights(path,naive)
    np.testing.assert_array_equal(naive.brain.motor.weights,trained.brain.motor.weights)
    np.testing.assert_array_equal(naive.brain.delta,trained.brain.delta)
    assert not naive.brain.voltage.any()
    naive.brain.motor.config['feature_groups']=16
    with pytest.raises(ValueError,match='configuration mismatch'):
        transfer_weights(path,naive)
