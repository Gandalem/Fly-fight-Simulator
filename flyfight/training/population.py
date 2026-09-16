from ..neural.plasticity import select_plastic
from ..agents.brain import FlyBrainState
from ..agents.fly import Fly
from ..agents.body import BodyCondition
from ..agents.internal_state import InternalState

def population(graph,config):
    subset=select_plastic(graph,config)
    return [Fly(i,FlyBrainState(graph,subset,config['seed']+i,config),BodyCondition(config),
                InternalState(config['body']['initial_energy'])) for i in range(config['training']['population'])]
