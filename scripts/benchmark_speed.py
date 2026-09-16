"""Paired old/new full-graph runs: exact state checks plus elapsed times.

Uses alternating run order; concurrent workloads can still affect timings.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
from flyfight.config import load_config
from flyfight.connectome.loader import load_connectome
from flyfight.training.population import population
from flyfight.training.checkpoint import save
from flyfight.embodiment.motor_adapter import MotorAdapter
from flyfight.embodiment.sensory_adapter import SensoryAdapter
from flyfight.embodiment.mujoco_env import MuJoCoArena
from flyfight.embodiment.cached_controller import CachedObservation
from flygym_demo.complex_terrain import HybridControllerObservation
from flyfight.environment.match import run_match
from flyfight.neural.lif import lif_step
from flyfight.neural.plasticity import update
from flyfight.runtime import Monitor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--duration',type=float,default=.3)
    p.add_argument('--pairs',type=int,default=2)
    p.add_argument('--output')
    args=p.parse_args()
    config=load_config('configs/quick.yaml'); config['arena']['duration']=args.duration
    graph=load_connectome(config['dataset']['processed'])
    out=Path(args.output or f'results/speed_{datetime.now():%Y%m%d_%H%M%S}')
    if out.exists(): raise ValueError(f'Output exists: {out}')
    out.mkdir(parents=True)
    records=[]; monitor=Monitor(config)
    for pair in range(args.pairs):
        paths={}; summaries={}; physics={}
        for optimized in ([False,True] if pair%2==0 else [True,False]):
            c=copy.deepcopy(config)
            c['runtime']['fast_neural']=optimized
            c['runtime']['brain_workers']=2 if optimized else 1
            c['arena']['controller_backend']='cached' if optimized else 'reference'
            pool=population(graph,c)
            # Warm compilation and reset all scientific/RNG state before timing.
            lif_step(pool[0].brain,np.zeros(graph.n,np.float32),c); update(pool[0].brain,0.,c)
            pool=population(graph,c)
            env=MuJoCoArena(c)
            try:
                observers=[CachedObservation(env.sim,f'fly{i}',env.controllers[i].legs) for i in range(2)]
                # Validate cached observer on live data before the paired match.
                for i,observer in enumerate(observers):
                    native=HybridControllerObservation.from_sim(env.sim,f'fly{i}')
                    cached=observer.read()
                    for field in vars(native): np.testing.assert_array_equal(getattr(native,field),getattr(cached,field))
                start=time.perf_counter()
                summary=run_match(env,pool,SensoryAdapter(graph,c),MotorAdapter(graph,c),c,1,713+pair)
                elapsed=time.perf_counter()-start
                label='optimized' if optimized else 'reference'
                checkpoint=out/f'{pair}_{label}.npz'
                save(checkpoint,pool,1,np.random.default_rng(9))
                paths[label]=checkpoint; summaries[label]=summary
                physics[label]={field:getattr(env.data,field).copy() for field in ('qpos','qvel','qacc','ctrl')}
                physics[label]['front_offsets']=env.front_offsets.copy()
                physics[label]['front_adhesion']=env.front_adhesion.copy()
                record=dict(pair=pair,backend=label,simulated_seconds=args.duration,wall_seconds=elapsed,timings=summary['timings'])
                records.append(record); print(json.dumps(record),flush=True)
            finally: env.close()
        assert summaries['reference']['agents']==summaries['optimized']['agents'], 'Behavior/learning differs'
        for field in physics['reference']:
            np.testing.assert_array_equal(physics['reference'][field],physics['optimized'][field],err_msg=field)
        with np.load(paths['reference']) as a,np.load(paths['optimized']) as b:
            for key in a.files: np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    reference=np.mean([r['wall_seconds'] for r in records if r['backend']=='reference'])
    optimized=np.mean([r['wall_seconds'] for r in records if r['backend']=='optimized'])
    report=dict(exact_scientific_state_match=True,exact_final_physics_state_match=True,
                records=records,speedup=float(reference/optimized),performance=monitor.sample())
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
