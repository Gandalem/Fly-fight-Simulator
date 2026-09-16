import copy
import numpy as np
from flyfight.agents.brain import FlyBrainState
from flyfight.neural.lif import lif_step
from flyfight.neural.plasticity import update
from flyfight.training.checkpoint import ARRAYS
from flyfight.embodiment.cached_controller import CachedHybridTurningController
from flygym_demo.complex_terrain import HybridTurningController, HybridControllerObservation, PreprogrammedSteps
from flygym_demo.complex_terrain.common import get_default_locomotion_dof_order
from test_neural import fixture


def test_compiled_neurons_and_plasticity_match_numpy_exactly():
    graph,plastic,config=fixture()
    config['neural'].update(noise_std=.5,reset_voltage=-.1)
    config['learning'].update(a_plus=.9,a_minus=1.13,max_relative_change=.3)
    slow=copy.deepcopy(config); slow['runtime']['fast_neural']=False
    fast=FlyBrainState(graph,plastic,12,config); reference=FlyBrainState(graph,plastic,12,slow)
    rng=np.random.default_rng(51)
    for step in range(300):
        drive=rng.uniform(0,2000,graph.n).astype(np.float32)
        reward=float(rng.uniform(-4,4))
        for brain,c in ((fast,config),(reference,slow)):
            lif_step(brain,drive,c); update(brain,reward,c,step%3!=0)
        for field in ARRAYS:
            np.testing.assert_array_equal(getattr(fast,field),getattr(reference,field),err_msg=field)
    assert fast.rng.bit_generator.state==reference.rng.bit_generator.state


def test_cached_controller_matches_native_across_reflexes_and_reverse():
    steps=PreprogrammedSteps(); order=get_default_locomotion_dof_order()
    controllers=[cls(timestep=.0001,preprogrammed_steps=steps,output_dof_order=order)
                 for cls in (HybridTurningController,CachedHybridTurningController)]
    for controller in controllers: controller.reset(seed=732)
    rng=np.random.default_rng(36)
    for tick in range(500):
        obs=HybridControllerObservation(.8,rng.uniform(.1,.8,6),rng.uniform(-5,5,(6,3,3)),np.array([1.,0,0]))
        signal=rng.uniform(-1,1,2)
        a,b=[controller.step(signal,obs) for controller in controllers]
        np.testing.assert_array_equal(a.joint_angles,b.joint_angles)
        np.testing.assert_array_equal(a.adhesion_onoff,b.adhesion_onoff)
        for field in controllers[0].last_info:
            np.testing.assert_array_equal(controllers[0].last_info[field],controllers[1].last_info[field])


def test_parallel_brains_preserve_each_state_and_rng():
    from flyfight.neural.stepper import BrainStepper
    graph,plastic,config=fixture()
    config['neural']['noise_std']=.2
    parallel=[FlyBrainState(graph,plastic,seed,config) for seed in (17,93)]
    serial=[FlyBrainState(graph,plastic,seed,config) for seed in (17,93)]
    single=copy.deepcopy(config); single['runtime']['brain_workers']=1
    config['runtime']['brain_workers']=2
    rng=np.random.default_rng(13)
    with BrainStepper(config,2) as multi, BrainStepper(single,2) as one:
        for tick in range(20):
            drives=[rng.uniform(0,2000,graph.n).astype(np.float32) for _ in range(2)]
            rewards=rng.uniform(-4,4,2)
            a=multi.advance(parallel,drives,rewards); b=one.advance(serial,drives,rewards)
            for i in range(2):
                np.testing.assert_array_equal(a[i],b[i])
                for field in ARRAYS:
                    np.testing.assert_array_equal(getattr(parallel[i],field),getattr(serial[i],field))
                assert parallel[i].rng.bit_generator.state==serial[i].rng.bit_generator.state
