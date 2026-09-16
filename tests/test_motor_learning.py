import numpy as np
import pytest
from flyfight.config import load_config
from flyfight.embodiment.motor_learning import MotorLearningState
from flyfight.neural.neuromodulation import homeostatic_signal


def test_delayed_reward_changes_selected_action_and_off_stays_fixed():
    config=load_config()
    trained=MotorLearningState(13,config)
    frozen=MotorLearningState(13,config)
    rates=np.ones(config['motor_learning']['feature_groups'])
    chosen=trained.decide(rates,.1)
    np.testing.assert_array_equal(chosen,frozen.decide(rates,.1,learn=False))
    features=trained.previous_features.copy()
    mean_before=trained.mean_action(features)
    # Reward arrives after an entire held action, not before its selection.
    for state in (trained,frozen):
        state.observe(0.,.05)
        state.observe(.2,.05)
    assert not trained.weights.any()
    trained.finish(); frozen.finish(learn=False)
    change=trained.mean_action(features)-mean_before
    sampled_latent=np.arctanh(chosen)
    assert np.dot(change,sampled_latent-mean_before)>0
    assert not frozen.weights.any() and not frozen.value_weights.any()
    assert trained.updates==1


def test_motor_reset_and_frozen_evaluation_do_not_erase_learning():
    config=load_config(); motor=MotorLearningState(3,config)
    motor.weights.fill(.2); motor.value_weights.fill(.1)
    motor.reset('B')
    assert np.all(motor.weights==np.float32(.2))
    motor.reset('C',.5)
    assert np.all(motor.weights==np.float32(.1))
    before=motor.weights.copy()
    action=motor.decide(np.zeros(32),.1,explore=False,learn=False)
    expected=np.tanh(motor.mean_action(motor.previous_features))
    np.testing.assert_allclose(action,expected)
    motor.observe(1.,.1); motor.finish(learn=False)
    np.testing.assert_array_equal(motor.weights,before)
    motor.reset('A')
    assert not motor.weights.any() and not motor.value_weights.any()


def test_homeostasis_penalizes_cost_and_injury_without_a_win_signal():
    config=load_config()
    assert homeostatic_signal(-.01,0,.1,config)<0
    assert homeostatic_signal(.01,0,.1,config)>0
    assert homeostatic_signal(0,0,.1,config)==0
    assert homeostatic_signal(.01,.1,.1,config)<0


def test_cns_and_motor_learning_can_be_ablated_independently():
    from flyfight.embodiment.motor_adapter import MotorAdapter
    # This property does not depend on a graph or a body.
    adapter=MotorAdapter.__new__(MotorAdapter)
    adapter.config=load_config()
    adapter.config['learning']['enabled']=False
    assert adapter.learning_enabled
    adapter.config['motor_learning']['plasticity']=False
    assert not adapter.learning_enabled
    adapter.config['motor_learning']['plasticity']=True
    adapter.config['training']['plasticity']=False
    assert not adapter.learning_enabled


@pytest.mark.parametrize('seed',[2,7,19])
def test_reward_learning_improves_held_out_actuator_calibration(seed):
    from flyfight.training.motor_benchmark import run_benchmark
    result=run_benchmark(seed,trials=1500)
    assert result['learned_mse']<result['frozen_mse']*.4
    assert result['reward_shuffled_mse']>result['learned_mse']*2
