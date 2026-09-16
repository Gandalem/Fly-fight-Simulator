import numpy as np
from flyfight.config import load_config
from flyfight.agents.body import BodyCondition
from flyfight.agents.internal_state import InternalState
from flyfight.combat.damage import apply_impact
from flyfight.combat.death import incapacity
from flyfight.environment.food import Food

def test_damage_integrates_time():
    c=load_config(); a=BodyCondition(c); b=BodyCondition(c)
    for _ in range(100): apply_impact(a,'FrontLegLeft',50,.001,c)
    for _ in range(200): apply_impact(b,'FrontLegLeft',50,.0005,c)
    assert abs(a.modifier('FrontLegLeft')-b.modifier('FrontLegLeft'))<1e-12
    a.parts['Head'].current_integrity=0
    assert incapacity(a,c)=='critical_head'

def test_hunger_and_exclusive_food():
    c=load_config(); states=[InternalState(),InternalState()]
    h=states[0].hunger; states[0].step(1,1,0,0,1,c); assert states[0].hunger>h
    food=Food(c); initial=food.amount
    intake,owner=food.consume(np.zeros((2,3)),states,.1,np.random.default_rng(0))
    assert (intake>0).sum()==1 and np.isclose(food.amount+intake.sum(),initial)
    h=states[owner].hunger; states[owner].step(.1,0,0,intake[owner],1,c)
    assert states[owner].hunger<h
