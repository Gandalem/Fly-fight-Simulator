"""Profile one short actual-graph match after loading data and warming Numba."""
from pathlib import Path
import cProfile
import pstats
import numpy as np
from flyfight.config import load_config
from flyfight.connectome.loader import load_connectome
from flyfight.training.population import population
from flyfight.embodiment.mujoco_env import MuJoCoArena
from flyfight.embodiment.motor_adapter import MotorAdapter
from flyfight.embodiment.sensory_adapter import SensoryAdapter
from flyfight.environment.match import run_match
from flyfight.neural.lif import lif_step
from flyfight.neural.plasticity import update

c=load_config('configs/quick.yaml'); c['arena']['duration']=.1
# cProfile does not attribute worker-thread calls; profile a single worker.
c['runtime']['brain_workers']=1
g=load_connectome(c['dataset']['processed']); pool=population(g,c)
motor=MotorAdapter(g,c); sensory=SensoryAdapter(g,c)
lif_step(pool[0].brain,np.zeros(g.n,np.float32),c); update(pool[0].brain,0.,c)
pool=population(g,c)
env=MuJoCoArena(c)
profile=cProfile.Profile()
try:
    profile.enable()
    run_match(env,pool,sensory,motor,c,1,713)
    profile.disable()
finally:
    env.close()
out=Path('results/hotspots'); out.mkdir(parents=True,exist_ok=True)
profile.dump_stats(str(out/'match.prof'))
with (out/'profile.txt').open('w',encoding='utf-8') as stream:
    pstats.Stats(profile,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(45)
print((out/'profile.txt').read_text(encoding='utf-8'))
