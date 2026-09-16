from datetime import datetime
from pathlib import Path
import json
import time
import numpy as np
from .population import population
from .matchmaking import choose
from .checkpoint import save,restore,transfer_weights
from ..embodiment.mujoco_env import MuJoCoArena
from ..embodiment.sensory_adapter import SensoryAdapter
from ..embodiment.motor_adapter import MotorAdapter
from ..embodiment.viewer import VideoRecorder,LiveViewer
from ..environment.match import run_match
from ..logging.episode_logger import EpisodeLogger
from ..runtime import Monitor,provenance

def train(graph,config,output=None,render=False,agents=2,resume=None,gui=False):
    c=config['training']
    if c['episodes']<1 or c['population']<agents: raise ValueError('Invalid episode/population count')
    # CPU event-driven backend keeps all neural arrays outside GPU memory.
    if config['runtime']['device']!='cpu':
        print('CUDA neural backend unavailable: falling back to CPU event-driven CSR.',flush=True)
    path=Path(output or f'results/run_{datetime.now():%Y%m%d_%H%M%S_%f}')
    if path.exists() and any(path.iterdir()): raise ValueError(f'Output already contains data: {path}; use a new output directory')
    path.mkdir(parents=True,exist_ok=True)
    (path/'run.json').write_text(json.dumps(provenance(config,graph),indent=2),encoding='utf-8')
    monitor=Monitor(config,path/'performance.jsonl')
    cached=graph.provenance.get('neural_weight_config')
    if cached and any(config['neural'][k]!=v for k,v in cached.items()):
        raise ValueError('Cached weights do not match neural config. Re-run prepare-connectome with this config.')
    pool=population(graph,config); rng=np.random.default_rng(config['seed'])
    if config.get('evaluation'):
        transfer_weights(config['evaluation']['checkpoint'],pool[0],config['evaluation']['agent'])
    start=restore(resume,pool,rng) if resume else 0
    sensory=SensoryAdapter(graph,config); motor=MotorAdapter(graph,config)
    (path/'mapping.json').write_text(json.dumps({'warning':'Unvalidated feature encoder and DN decoder hypotheses',
                                               'sensory':sensory.describe(),'motor':motor.describe()},indent=2))
    save(path/'initial.npz',pool,start,rng)
    save(path/'checkpoint.npz',pool,start,rng)
    logger=EpisodeLogger(path)
    env=None; video=None; viewer=None; completed=start; total_steps=0; last_profile=time.monotonic()
    try:
        env=MuJoCoArena(config,agents)
        monitor.sample()
        if render: video=VideoRecorder(path/'fight.mp4',config['arena']['render_fps'])
        if gui: viewer=LiveViewer(env)
        def display(frame,flies,episode,t):
            if video: video(frame,flies,episode,t)
            if viewer: viewer(frame,flies,episode,t)
        def progress(steps):
            nonlocal last_profile
            if time.monotonic()-last_profile>=10:
                monitor.sample(total_steps+steps,completed-start)
                last_profile=time.monotonic()
        for episode in range(start+1,start+c['episodes']+1):
            if config.get('evaluation'):
                flies=[pool[0],pool[1+int(rng.integers(len(pool)-1))]]
                if rng.random()<.5: flies.reverse()
            else:
                flies=choose(pool,agents,rng,c['random_matchmaking'])
            seed=int(rng.integers(0,2**31))
            t=time.perf_counter()
            summary=run_match(env,flies,sensory,motor,config,episode,seed,display if video or viewer else None,progress)
            summary.update(wall_seconds=time.perf_counter()-t,plasticity_on=c['plasticity'],reset_mode=c['reset_mode'])
            logger.write(episode,summary); completed=episode
            if video: video.finish(summary)
            if viewer: viewer.finish(summary)
            total_steps+=summary['neural_steps']
            if episode%config['runtime']['profile_every']==0 or episode==start+1:
                perf=monitor.sample(total_steps,episode-start)
                print(json.dumps({'episode':episode,'winner':summary['winner'],'food':[s['food_consumed'] for s in summary['agents']],
                                  'ram_gib':perf['ram_gib'],'neural_steps_per_sec':perf['steps_per_sec']}),flush=True)
            if episode%c['checkpoint_every']==0: save(path/'checkpoint.npz',pool,episode,rng)
        save(path/'checkpoint.npz',pool,completed,rng)
        (path/'complete.json').write_text(json.dumps(dict(episodes=completed-start,performance=monitor.sample(total_steps,completed-start)),indent=2))
    except (MemoryError,KeyboardInterrupt) as error:
        # Last committed checkpoint remains valid; don't label a partial match complete.
        (path/'interrupted.json').write_text(json.dumps({'last_completed_episode':completed,'reason':str(error)}))
        print(f'Run interrupted safely; last checkpoint: {path / "checkpoint.npz"}',flush=True)
    finally:
        logger.close()
        if video: video.close()
        if viewer: viewer.close()
        if env: env.close()
    return str(path)
