import numpy as np
import pytest
from flyfight.config import load_config
from flyfight.embodiment.mujoco_env import MuJoCoArena
from flyfight.agents.body import BodyCondition
from flyfight.agents.internal_state import InternalState

@pytest.mark.physics
def test_real_contacts_damage_actuators_and_reset():
    c=load_config(); c['arena']['starting_distance']=3.8
    c['body']['damage_force_threshold']=0.; c['body']['damage_multiplier']=.02
    env=MuJoCoArena(c)
    try:
        bodies=[BodyCondition(c),BodyCondition(c)]; internal=[InternalState(),InternalState()]
        contacts=0; damage=0
        for _ in range(30):
            result=env.step([{'legs':[1,1]}]*2,bodies,internal)
            contacts+=len(result['contacts']); damage+=sum(result['damage_received'])
        assert contacts>0 and damage>0
        # Independent foreleg targets reach the real position actuators and
        # stay within joint ranges; no gesture/attack sequence is executed.
        before=env.data.qpos.copy()
        command={'legs':[.5,.5],'front_joints':[[1.,-.5,1.],[-1.,.5,-1.]],'front_adhesion':[0.,1.]}
        for _ in range(5): env.step([command,{'legs':[.5,.5]}],bodies,internal)
        assert np.linalg.norm(env.front_offsets[0])>.1
        assert env.front_adhesion[0,0]<.3 and env.front_adhesion[0,1]==1.
        assert np.max(np.abs(env.data.qpos-before))>.01
        assert np.isfinite(env.data.qpos).all()
        bodies[0].parts['FrontLegLeft'].current_integrity=0
        env.step([{'legs':[1,1]}]*2,bodies,internal)
        selected=[k for k,(i,p) in env.actuator_part.items() if i==0 and p=='FrontLegLeft']
        assert selected and np.all(env.model.actuator_gainprm[selected]==0)
        env.reset(42)
        np.testing.assert_array_equal(env.model.actuator_gainprm,env.base_gain)
        assert not env.front_offsets.any() and np.all(env.front_adhesion==1)
        assert np.isfinite(env.data.qpos).all()
    finally: env.close()
