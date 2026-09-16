import json
import numpy as np
from flyfight.config import load_config
from flyfight.embodiment.mujoco_env import MuJoCoArena
from flyfight.agents.body import BodyCondition
from flyfight.agents.internal_state import InternalState
from flyfight.runtime import Monitor

c=load_config(); monitor=Monitor(c)
env=MuJoCoArena(c,2)
initial=env.positions()
bodies=[BodyCondition(c) for _ in range(2)]
internal=[InternalState() for _ in range(2)]
count=0
for i in range(50):
    obs=env.step([{'legs':[1,1]}]*2,bodies,internal)
    count+=len(obs['contacts'])
print(json.dumps({'initial':initial.tolist(),'final':env.positions().tolist(),'contacts':count,'joints':env.model.njnt,'actuators':env.model.nu,'memory':monitor.sample(5000)},indent=2))
from PIL import Image
Image.fromarray(env.frame()).save('results/physics_smoke.png')
env.close()
