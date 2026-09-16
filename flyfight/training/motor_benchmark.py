"""Synthetic actuator-credit diagnostic. Not an embodied or fighting experiment."""
import numpy as np
from ..config import load_config
from ..embodiment.motor_learning import MotorLearningState


def run_benchmark(seed, trials=1500):
    config=load_config()
    states={name:MotorLearningState(seed,config) for name in ('learned','frozen','reward_shuffled')}
    rng=np.random.default_rng(seed+91)
    # Unknown targets in a toy calibration task; never supplied to the policy.
    targets=np.array([[-.6,.55],[.6,-.55]])
    rates=np.zeros((2,config['motor_learning']['feature_groups']))
    rates[0,0]=1.; rates[1,1]=1.
    for trial in range(trials):
        context=int(rng.integers(2))
        wrong_context=int(rng.integers(2))
        for name,state in states.items():
            state.reset('B')
            action=state.decide(rates[context],.1,learn=name!='frozen')
            goal=targets[wrong_context if name=='reward_shuffled' else context]
            # Bounded scalar actuator accuracy; absent from the real arena reward.
            reward=-float(np.mean((action[:2]-goal)**2))
            state.observe(0.,.05)
            state.observe(reward,.05)
            state.finish(learn=name!='frozen')
    result=dict(seed=seed,trials=trials,task='synthetic_actuator_calibration_not_combat')
    # Fresh rollouts with exploration and learning off, same two contexts.
    for name,state in states.items():
        errors=[]
        for context in range(2):
            state.reset('B')
            action=state.decide(rates[context],.1,explore=False,learn=False)
            errors.append(float(np.mean((action[:2]-targets[context])**2)))
            state.finish(learn=False)
        result[name+'_mse']=float(np.mean(errors))
    return result
