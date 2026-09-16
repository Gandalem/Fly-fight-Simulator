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
        bodies[0].parts['FrontLegLeft'].current_integrity=0
        env.step([{'legs':[1,1]}]*2,bodies,internal)
        selected=[k for k,(i,p) in env.actuator_part.items() if i==0 and p=='FrontLegLeft']
        assert selected and np.all(env.model.actuator_gainprm[selected]==0)
        env.reset(42)
        np.testing.assert_array_equal(env.model.actuator_gainprm,env.base_gain)
        assert np.isfinite(env.data.qpos).all()
    finally: env.close()
